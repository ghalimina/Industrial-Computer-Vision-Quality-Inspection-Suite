@echo off
chcp 65001 >nul
title رفع المستودع إلى GitHub
echo ================================================================
echo       🚀 رفع مشروع الرؤية الحاسوبية الصناعية إلى GitHub
echo       Repository: Industrial-Computer-Vision-Quality-Inspection-Suite
echo ================================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ================================================================
    echo       🎉 تم الرفع بنجاح إلى GitHub!
    echo       🌐 الرابط: https://github.com/ghalimina/Industrial-Computer-Vision-Quality-Inspection-Suite
    echo ================================================================
) else (
    echo ⚠️ تأكد من النقر على زر "Create repository" أولاً في المتصفح، ثم أعد المحاولة.
)
pause
