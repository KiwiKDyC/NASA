param(
    [Parameter(Position=0)]
    [string]$Mensaje = "chore: actualizacion automatica del proyecto AgroRotacion NASA"
)

# Asegurar que Git este en el PATH de la sesion actual
$gitCmdPath = "C:\Program Files\Git\cmd"
if ($env:PATH -notlike "*$gitCmdPath*") {
    $env:PATH += ";$gitCmdPath"
}

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  SINCRONIZANDO REPOSITORIO CON GITHUB (origin/main)" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 1. Traer posibles cambios del equipo antes de subir (evita conflictos)
Write-Host "[1/3] Verificando cambios remotos del equipo (git pull --rebase)..." -ForegroundColor Yellow
git pull origin main --rebase 2>$null

# 2. Agregar archivos nuevos o modificados (respetando .gitignore)
Write-Host "[2/3] Preparando cambios locales (git add .)..." -ForegroundColor Yellow
git add .

$estado = git status --porcelain
if ([string]::IsNullOrWhiteSpace($estado)) {
    Write-Host "  -> No hay cambios locales nuevos por confirmar (commit)." -ForegroundColor Gray
} else {
    git commit -m "$Mensaje"
    Write-Host "  -> Commit creado: '$Mensaje'" -ForegroundColor Green
}

# 3. Subir a GitHub
Write-Host "[3/3] Subiendo a GitHub (git push -u origin main)..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[OK] Repositorio sincronizado exitosamente en https://github.com/KiwiKDyC/NASA" -ForegroundColor Green
} else {
    Write-Host "`n[!] No se pudo completar el push. Verifica que 'itskiwik' tenga permisos de escritura en https://github.com/KiwiKDyC/NASA o que hayas iniciado sesion con la cuenta correcta." -ForegroundColor Red
}
