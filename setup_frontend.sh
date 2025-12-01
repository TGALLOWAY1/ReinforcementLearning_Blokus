#!/bin/bash

# Blokus RL Frontend Setup Script

echo "🎮 Setting up Blokus RL Frontend..."
echo "=================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed!"
    echo "Please install Node.js from https://nodejs.org/"
    echo "Or use a package manager:"
    echo "  macOS: brew install node"
    echo "  Ubuntu: sudo apt install nodejs npm"
    echo "  Windows: choco install nodejs"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed!"
    echo "Please install npm (usually comes with Node.js)"
    exit 1
fi

echo "✅ npm found: $(npm --version)"

# Navigate to frontend directory
cd frontend

echo "📦 Installing dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo ""
    echo "🚀 To start the frontend:"
    echo "   cd frontend"
    echo "   npm run dev"
    echo ""
    echo "🌐 The frontend will be available at: http://localhost:5173"
    echo ""
    echo "📋 Make sure the backend is running:"
    echo "   python3 run_server.py"
    echo ""
    echo "🎯 Then open your browser to start playing!"
else
    echo "❌ Failed to install dependencies"
    echo "Please check the error messages above and try again"
    exit 1
fi
