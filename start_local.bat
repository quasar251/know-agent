@echo off
chcp 65001 >nul
setlocal
title know 本地一键启动

cd /d "%~dp0"
set "ROOT=%~dp0"

echo ============================================
echo   know 本地启动（非 Docker）
echo   后端 :8000   ^|  前端 :3000   ^|  Redis :6379
echo ============================================
echo.

REM ---------- 1) Redis（短期记忆 hot cache；缺失则自动退化为纯 PG）----------
REM REDIS_DIR 可用环境变量覆盖；未设置时自动在 PATH 上查找 redis-server.exe。
REM 注意：短期记忆的总开关是 backend\.env 里的 REDIS_URL，这里只负责把服务拉起来。
if not defined REDIS_DIR (
  for %%I in (redis-server.exe) do set "REDIS_DIR=%%~dp$PATH:I"
)
if not defined REDIS_DIR if exist redis-server.exe set "REDIS_DIR=%CD%\"

netstat -ano | findstr /c:":6379 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo [1/5] Redis 已在 :6379 运行，跳过。
) else if not defined REDIS_DIR (
  echo [1/5] 未找到 redis-server.exe —— 短期记忆将退化为纯 PG 路径（功能可用，只是没有 hot cache）。
  echo        要启用请把 redis-server.exe 加入 PATH，或用 REDIS_DIR 指定其所在目录。
) else if exist "%REDIS_DIR%redis.conf" (
  echo [1/5] 启动 Redis :6379 ...
  start "know redis :6379" /min /d "%REDIS_DIR%" redis-server.exe redis.conf
) else (
  echo [1/5] 启动 Redis :6379 ...
  start "know redis :6379" /min /d "%REDIS_DIR%" redis-server.exe
)

REM ---------- 2) 后端依赖 ----------
if exist "backend\.venv\Scripts\python.exe" (
  echo [2/5] 后端虚拟环境已存在，跳过依赖安装。
) else (
  echo [2/5] 未找到 backend\.venv，正在创建虚拟环境并安装依赖（首次较慢）...
  pushd backend
  python -m venv .venv
  if errorlevel 1 (
    echo [错误] 创建虚拟环境失败。请确认已安装 Python 3.11+ 并加入 PATH。
    popd & pause & exit /b 1
  )
  call ".venv\Scripts\python.exe" -m pip install --upgrade pip
  call ".venv\Scripts\python.exe" -m pip install -e ".[milvus]"
  if errorlevel 1 (
    echo [错误] 后端依赖安装失败。
    popd & pause & exit /b 1
  )
  popd
)

REM ---------- 3) 后端配置 ----------
if exist "backend\.env" (
  echo [3/5] 后端 .env 已就绪。
) else (
  echo [3/5] 未找到 backend\.env，已从 env.example 复制一份。
  copy "backend\env.example" "backend\.env" >nul
  echo        请填写 LLM / Embedding 的 API Key 后重新运行本脚本。
)

REM ---------- 4) 启动后端 ----------
echo [4/5] 启动后端 http://127.0.0.1:8000 ...
start "know backend :8000" /d "%ROOT%backend" cmd /k ".venv\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8000"

REM ---------- 5) 前端依赖 + 启动 ----------
if exist "frontend\node_modules" (
  echo [5/5] 前端依赖已存在，直接启动 http://localhost:3000 ...
) else (
  echo [5/5] 未找到 frontend\node_modules，正在安装依赖（首次较慢）...
  pushd frontend
  call npm install
  if errorlevel 1 (
    echo [错误] npm install 失败。请确认已安装 Node.js 20+。
    popd & pause & exit /b 1
  )
  popd
)
start "know frontend :3000" /d "%ROOT%frontend" cmd /k "npm run dev"

echo.
echo 已启动完成：
echo   后端 http://127.0.0.1:8000
echo   前端 http://localhost:3000   ^(浏览器打开这个^)
echo   Redis :6379   ^(短期记忆 hot cache^)
echo.
echo 关闭服务：直接关掉弹出的命令行窗口即可（Redis 窗口仅在本脚本拉起它时才存在）。
echo.
pause
endlocal