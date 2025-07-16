#!/bin/bash
# Script para iniciar el pipeline de trading con monitoreo automático
# =================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
PROJECT_ID="trading-ai-460823"
NAMESPACE="trading"
LOG_ANALYZER_CONFIG="log_analyzer_config.yaml"

# Función para logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Función para verificar dependencias
check_dependencies() {
    log_info "Verificando dependencias..."
    
    # Verificar kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl no está instalado"
        exit 1
    fi
    
    # Verificar gcloud
    if ! command -v gcloud &> /dev/null; then
        log_warning "gcloud no está instalado - algunas funciones pueden no funcionar"
    fi
    
    # Verificar Python
    if ! command -v python &> /dev/null; then
        log_error "Python no está instalado"
        exit 1
    fi
    
    log_success "Dependencias verificadas"
}

# Función para verificar configuración de GCP
check_gcp_config() {
    log_info "Verificando configuración de GCP..."
    
    # Verificar autenticación
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_warning "No hay autenticación activa de GCP"
        log_info "Ejecutando: gcloud auth application-default login"
        gcloud auth application-default login
    fi
    
    # Verificar proyecto
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
        log_info "Configurando proyecto: $PROJECT_ID"
        gcloud config set project $PROJECT_ID
    fi
    
    log_success "Configuración de GCP verificada"
}

# Función para verificar namespace
check_namespace() {
    log_info "Verificando namespace: $NAMESPACE"
    
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_info "Creando namespace: $NAMESPACE"
        kubectl create namespace $NAMESPACE
    fi
    
    log_success "Namespace verificado"
}

# Función para iniciar el analizador de logs
start_log_analyzer() {
    log_info "Iniciando analizador de logs..."
    
    # Verificar si ya está ejecutándose
    if kubectl get deployment log-analyzer -n $NAMESPACE &> /dev/null; then
        log_info "Analizador de logs ya está desplegado"
        
        # Verificar si está listo
        if kubectl get deployment log-analyzer -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' | grep -q "1"; then
            log_success "Analizador de logs está ejecutándose"
            return 0
        else
            log_warning "Analizador de logs no está listo, esperando..."
            kubectl wait --for=condition=available --timeout=300s deployment/log-analyzer -n $NAMESPACE
        fi
    else
        # Desplegar analizador de logs
        log_info "Desplegando analizador de logs..."
        
        # Aplicar deployment
        kubectl apply -f k8s/log-analyzer-deployment.yaml
        
        # Esperar a que esté listo
        log_info "Esperando a que el analizador esté listo..."
        kubectl wait --for=condition=available --timeout=300s deployment/log-analyzer -n $NAMESPACE
        
        log_success "Analizador de logs desplegado y ejecutándose"
    fi
}

# Función para iniciar el pipeline de trading
start_trading_pipeline() {
    log_info "Iniciando pipeline de trading..."
    
    # Verificar si ya está ejecutándose
    if kubectl get deployment trading-pipeline -n $NAMESPACE &> /dev/null; then
        log_info "Pipeline de trading ya está desplegado"
        
        # Verificar si está listo
        if kubectl get deployment trading-pipeline -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' | grep -q "1"; then
            log_success "Pipeline de trading está ejecutándose"
            return 0
        else
            log_warning "Pipeline de trading no está listo, esperando..."
            kubectl wait --for=condition=available --timeout=300s deployment/trading-pipeline -n $NAMESPACE
        fi
    else
        # Desplegar pipeline de trading
        log_info "Desplegando pipeline de trading..."
        
        # Aplicar deployment del pipeline (ajustar según tu configuración)
        kubectl apply -f k8s/trading-pipeline-deployment.yaml
        
        # Esperar a que esté listo
        log_info "Esperando a que el pipeline esté listo..."
        kubectl wait --for=condition=available --timeout=300s deployment/trading-pipeline -n $NAMESPACE
        
        log_success "Pipeline de trading desplegado y ejecutándose"
    fi
}

# Función para mostrar logs
show_logs() {
    log_info "Mostrando logs del sistema..."
    
    echo -e "\n${BLUE}=== Logs del Pipeline de Trading ===${NC}"
    kubectl logs -f deployment/trading-pipeline -n $NAMESPACE &
    TRADING_LOGS_PID=$!
    
    echo -e "\n${BLUE}=== Logs del Analizador ===${NC}"
    kubectl logs -f deployment/log-analyzer -n $NAMESPACE &
    ANALYZER_LOGS_PID=$!
    
    # Función para limpiar procesos al salir
    cleanup() {
        log_info "Deteniendo logs..."
        kill $TRADING_LOGS_PID 2>/dev/null || true
        kill $ANALYZER_LOGS_PID 2>/dev/null || true
        exit 0
    }
    
    trap cleanup SIGINT SIGTERM
    
    # Esperar a que ambos procesos terminen
    wait
}

# Función para mostrar estado del sistema
show_status() {
    log_info "Estado del sistema:"
    
    echo -e "\n${BLUE}=== Pods ===${NC}"
    kubectl get pods -n $NAMESPACE
    
    echo -e "\n${BLUE}=== Servicios ===${NC}"
    kubectl get services -n $NAMESPACE
    
    echo -e "\n${BLUE}=== Deployments ===${NC}"
    kubectl get deployments -n $NAMESPACE
}

# Función para detener el sistema
stop_system() {
    log_info "Deteniendo sistema..."
    
    # Detener analizador de logs
    if kubectl get deployment log-analyzer -n $NAMESPACE &> /dev/null; then
        log_info "Deteniendo analizador de logs..."
        kubectl delete deployment log-analyzer -n $NAMESPACE
    fi
    
    # Detener pipeline de trading
    if kubectl get deployment trading-pipeline -n $NAMESPACE &> /dev/null; then
        log_info "Deteniendo pipeline de trading..."
        kubectl delete deployment trading-pipeline -n $NAMESPACE
    fi
    
    log_success "Sistema detenido"
}

# Función principal
main() {
    echo -e "${GREEN}🚀 Iniciando Sistema de Trading con Monitoreo${NC}"
    echo "=================================================="
    
    # Parsear argumentos
    case "${1:-start}" in
        "start")
            check_dependencies
            check_gcp_config
            check_namespace
            start_log_analyzer
            start_trading_pipeline
            show_status
            log_success "Sistema iniciado correctamente"
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "stop")
            stop_system
            ;;
        "restart")
            stop_system
            sleep 5
            main start
            ;;
        *)
            echo "Uso: $0 {start|logs|status|stop|restart}"
            echo ""
            echo "Comandos:"
            echo "  start   - Iniciar el sistema completo"
            echo "  logs    - Mostrar logs en tiempo real"
            echo "  status  - Mostrar estado del sistema"
            echo "  stop    - Detener el sistema"
            echo "  restart - Reiniciar el sistema"
            exit 1
            ;;
    esac
}

# Ejecutar función principal
main "$@" 