@echo off
echo =====================================================================
echo  Food Adulteration Risk - GitHub Remote Deployment Setup
echo =====================================================================
echo.
echo Step 1: Checking current git remote...
git remote -v
echo.
echo Step 2: Pushing 'main' branch to GitHub...
git push -u origin main
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =====================================================================
    echo [ERROR / ACTION REQUIRED]
    echo If you see "Repository not found", please ensure you have created
    echo the repository on GitHub:
    echo.
    echo 1. Open your browser to: https://github.com/new
    echo 2. Repository Name: food-adulteration-risk
    echo 3. Visibility: Public (recommended)
    echo 4. Do NOT initialize with README, .gitignore, or License!
    echo 5. Click "Create repository"
    echo 6. Re-run this script: deploy_to_github.bat
    echo =====================================================================
) else (
    echo.
    echo =====================================================================
    echo [SUCCESS] Repository pushed to GitHub successfully!
    echo Now open: https://share.streamlit.io/
    echo Or click the "Deploy" button in your Streamlit app header!
    echo =====================================================================
)
pause
