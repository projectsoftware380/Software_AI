# =========================================================================
# FUNCIÓN PARA ELIMINAR IMÁGENES EXISTENTES
# =========================================================================
function Remove-ExistingImages {
    param(
        [string]$ProjectID,
        [string]$Region,
        [string]$RepoName,
        [string]$ImageName
    )

    $RepoUrl = "${Region}-docker.pkg.dev/${ProjectID}/${RepoName}"

    Write-Host "--------------------------------------------------"
    Write-Host "Paso 0: Buscando y eliminando imágenes existentes en '$RepoUrl'..."
    Write-Host "--------------------------------------------------"

    # Comando para listar todas las imágenes, incluyendo sus etiquetas y digest
    $images = gcloud artifacts docker images list $RepoUrl --filter="package~${ImageName}" --format="get(image)" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "¡Falló la obtención de la lista de imágenes de Artifact Registry!"
        # Se decide continuar aunque falle, para no bloquear el pipeline si solo es un problema de listado
        return
    }

    if ($images) {
        # Convertir la salida en un array de líneas
        $imageList = $images -split "`n" | ForEach-Object { $_.Trim() }

        foreach ($imageUri in $imageList) {
            if ($imageUri) {
                Write-Host "Eliminando imagen: $imageUri"
                # Se usa --quiet para evitar la confirmación interactiva (y/n)
                gcloud artifacts docker images delete $imageUri --quiet
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "No se pudo eliminar la imagen: $imageUri. Puede que ya no exista o haya un problema de permisos."
                }
            }
        }
        Write-Host "Limpieza de imágenes completada."
    } else {
        Write-Host "No se encontraron imágenes existentes para eliminar."
    }
}


# --- 1. Configuración ---
$ProjectID = "trading-ai-460823"
$RepoName = "data-ingestion-repo"
$ImageName = "data-ingestion-agent"
$Region = "europe-west1"

# --- Llamada a la función de limpieza ---
Remove-ExistingImages -ProjectID $ProjectID -Region $Region -RepoName $RepoName -ImageName $ImageName


# --- 2. Generar Etiqueta Única ---
$VersionTag = Get-Date -Format "yyyyMMdd-HHmmss"
$ImageUri = "${Region}-docker.pkg.dev/${ProjectID}/${RepoName}/${ImageName}:${VersionTag}"

Write-Host "--------------------------------------------------"
Write-Host "Paso 1: Usando la URI de la imagen: $ImageUri"
Write-Host "--------------------------------------------------"


# --- 3. Construir y Subir la Imagen Docker ---
Write-Host "Paso 2: Construyendo la imagen Docker..."
# NOTA: Se añade --no-cache para forzar la reinstalación de las dependencias
# y evitar problemas si requirements.txt cambió.
docker build --no-cache -t $ImageUri .
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

# =========================================================================
# === AJUSTE CRÍTICO: Usar la ruta explícita al Python del entorno venv ===
# Esto garantiza que se use el intérprete correcto con las librerías
# instaladas (kfp, google-cloud-aiplatform, etc.) y evita el error
# 'ModuleNotFoundError' en la máquina local.
.\venv\Scripts\python.exe -m src.pipeline.main --common-image-uri $ImageUri
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