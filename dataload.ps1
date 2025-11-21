# init_env.ps1 파일 내용
$items = @("dataset", "mock_data", "models", "result")
$driveBase = "G:\내 드라이브\academytool-ai-db"

foreach ($item in $items) {
    if (Test-Path ".\$item") { Remove-Item ".\$item" -Recurse -Force }
    New-Item -ItemType SymbolicLink -Path ".\$item" -Target "$driveBase\$item" -Force
}
Write-Host "환경 설정 완료!"