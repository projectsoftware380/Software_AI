# run_pipeline.ps1

# --- 1. Configuración ---
$ProjectID = "trading-ai-460823"
$RepoName = "data-ingestion-repo"
$ImageName = "data-ingestion-agent"
$Region = "europe-west1"

# --- 2. Generar Etiqueta Única ---
$VersionTag = Get-Date -Format "yyyyMMdd-HHmmss"
$ImageUri = "${Region}-docker.pkg.dev/${ProjectID}/${RepoName}/${ImageName}:${VersionTag}"

Write-Host "--------------------------------------------------"
Write-Host "Paso 1: Usando la URI de la imagen: $ImageUri"
Write-Host "--------------------------------------------------"


# --- 3. Construir y Subir la Imagen Docker ---
Write-Host "Paso 2: Construyendo la imagen Docker..."
docker build -t $ImageUri .
if ($LASTEXITCODE -ne 0) {
    Write-Error "¡Falló la construcción de la imagen Docker!"
    exit 1
}

Write-Host "Paso 3: Subiendo la imagen a Artifact Registry..."
docker push $ImageUri
if ($LASTEXITCODE -ne 0) {
    Write-Error "¡Falló la subida de la imagen a Artifact Registry!"
    exit 1
}


# --- 4. Ejecutar la Pipeline Pasando la URI como Parámetro ---
Write-Host "Paso 4: Lanzando la pipeline de Vertex AI..."

# === AJUSTE CORREGIDO: Se cambia '--image_uri' por '--common-image-uri' ===
python -m src.pipeline.main --common-image-uri $ImageUri
# =========================================================================

# Verificación final
if ($LASTEXITCODE -eq 0) {
    Write-Host "--------------------------------------------------"
    Write-Host "¡Pipeline lanzada con éxito con la imagen $VersionTag!"
    Write-Host "--------------------------------------------------"
} else {
    Write-Error "¡Falló el lanzamiento de la pipeline!"
    exit 1
}