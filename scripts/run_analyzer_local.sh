#!/bin/bash
# Script para ejecutar el analizador de logs localmente
# ====================================================
# Sin costos de GCP - Solo usa las APIs gratuitas de logging

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    
    if ! command -v python &> /dev/null; then
        log_error "Python no está instalado"
        exit 1
    fi
    
    # Verificar si existen los archivos necesarios
    if [ ! -f "src/log_analyzer/main.py" ]; then
        log_error "No se encontró el analizador de logs"
        exit 1
    fi
    
    if [ ! -f "requirements-analyzer.txt" ]; then
        log_error "No se encontró requirements-analyzer.txt"
        exit 1
    fi
    
    log_success "Dependencias verificadas"
}

# Función para instalar dependencias
install_dependencies() {
    log_info "Instalando dependencias..."
    
    if [ ! -d "venv" ]; then
        log_info "Creando entorno virtual..."
        python -m venv venv
    fi
    
    # Activar entorno virtual
    source venv/bin/activate
    
    # Instalar dependencias
    pip install -r requirements-analyzer.txt
    
    log_success "Dependencias instaladas"
}

# Función para configurar GCP (solo autenticación gratuita)
setup_gcp_auth() {
    log_info "Configurando autenticación de GCP..."
    
    # Verificar si ya está autenticado
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_warning "No hay autenticación activa de GCP"
        log_info "Ejecutando autenticación gratuita..."
        gcloud auth application-default login
    fi
    
    # Configurar proyecto
    gcloud config set project trading-ai-460823
    
    log_success "Autenticación GCP configurada"
}

# Función para ejecutar analizador
run_analyzer() {
    log_info "Iniciando analizador de logs local..."
    
    # Activar entorno virtual
    source venv/bin/activate
    
    # Ejecutar analizador con configuración local
    python src/log_analyzer/main.py \
        --project-id=trading-ai-460823 \
        --verbose \
        --severity-threshold=INFO
}

# Función para ejecutar en modo daemon
run_daemon() {
    log_info "Iniciando analizador en modo daemon..."
    
    # Crear archivo de log
    LOG_FILE="logs/analyzer_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p logs
    
    # Ejecutar en background
    nohup python src/log_analyzer/main.py \
        --project-id=trading-ai-460823 \
        --verbose \
        --severity-threshold=INFO > "$LOG_FILE" 2>&1 &
    
    ANALYZER_PID=$!
    echo $ANALYZER_PID > logs/analyzer.pid
    
    log_success "Analizador iniciado en background (PID: $ANALYZER_PID)"
    log_info "Log file: $LOG_FILE"
    log_info "Para detener: $0 stop"
    log_info "Para ver logs: tail -f $LOG_FILE"
}

# Función para detener analizador
stop_analyzer() {
    log_info "Deteniendo analizador..."
    
    if [ -f "logs/analyzer.pid" ]; then
        PID=$(cat logs/analyzer.pid)
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            log_success "Analizador detenido (PID: $PID)"
        else
            log_warning "Analizador no estaba ejecutándose"
        fi
        rm -f logs/analyzer.pid
    else
        log_warning "No se encontró archivo PID"
    fi
}

# Función para mostrar estado
show_status() {
    log_info "Estado del analizador:"
    
    if [ -f "logs/analyzer.pid" ]; then
        PID=$(cat logs/analyzer.pid)
        if kill -0 $PID 2>/dev/null; then
            echo -e "${GREEN}✅ Analizador ejecutándose (PID: $PID)${NC}"
            
            # Mostrar logs recientes
            echo -e "\n${BLUE}=== Logs Recientes ===${NC}"
            tail -n 20 logs/analyzer_*.log 2>/dev/null || echo "No hay logs disponibles"
        else
            echo -e "${RED}❌ Analizador no está ejecutándose${NC}"
            rm -f logs/analyzer.pid
        fi
    else
        echo -e "${YELLOW}⚠️  Analizador no está ejecutándose${NC}"
    fi
}

# Función para mostrar logs en tiempo real
show_logs() {
    log_info "Mostrando logs en tiempo real..."
    
    if [ -f "logs/analyzer.pid" ]; then
        PID=$(cat logs/analyzer.pid)
        if kill -0 $PID 2>/dev/null; then
            # Encontrar el archivo de log más reciente
            LATEST_LOG=$(ls -t logs/analyzer_*.log 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                tail -f "$LATEST_LOG"
            else
                log_error "No se encontraron archivos de log"
            fi
        else
            log_error "Analizador no está ejecutándose"
        fi
    else
        log_error "No se encontró archivo PID"
    fi
}

# Función para mostrar ayuda
show_help() {
    echo -e "${BLUE}Uso: $0 [COMANDO]${NC}"
    echo ""
    echo "Comandos disponibles:"
    echo "  start     - Iniciar analizador (modo interactivo)"
    echo "  daemon    - Iniciar analizador en background"
    echo "  stop      - Detener analizador"
    echo "  status    - Mostrar estado del analizador"
    echo "  logs      - Mostrar logs en tiempo real"
    echo "  install   - Instalar dependencias"
    echo "  help      - Mostrar esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  $0 install    # Primera vez: instalar dependencias"
    echo "  $0 daemon     # Iniciar en background"
    echo "  $0 logs       # Ver logs en tiempo real"
    echo "  $0 stop       # Detener analizador"
    echo ""
    echo "💡 Recomendación: Usa 'daemon' para ejecutar continuamente"
}

# Función principal
main() {
    case "${1:-help}" in
        "start")
            check_dependencies
            setup_gcp_auth
            run_analyzer
            ;;
        "daemon")
            check_dependencies
            setup_gcp_auth
            run_daemon
            ;;
        "stop")
            stop_analyzer
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "install")
            check_dependencies
            install_dependencies
            setup_gcp_auth
            log_success "Instalación completada"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Comando desconocido: $1"
            show_help
            exit 1
            ;;
    esac
}

# Ejecutar función principal
main "$@" 