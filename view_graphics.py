#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphics Viewer Server for PINN Comparison
"""

import http.server
import socketserver
import os
import webbrowser
import time
from pathlib import Path

# Graphics file path
GRAPHICS_DIR = '/home/mehmetcelik/Masaüstü/masaüstü/makale/finance ml'
GRAPHICS_FILE = 'model_comparison_final.png'
GRAPHICS_PATH = os.path.join(GRAPHICS_DIR, GRAPHICS_FILE)

PORT = 8888

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Main page
            html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PINN vs MLP Comparison</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }
        h2 {
            color: #0066cc;
            margin-top: 30px;
        }
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #0066cc;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        .results {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #0066cc;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .graph-container {
            text-align: center;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .graph-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        .highlight {
            background-color: #fff59d;
            padding: 2px 6px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>[SUCCESS] PINN vs MLP - Model Comparison</h1>
    <h1>Physics-Informed Neural Networks vs Multi-Layer Perceptron</h1>
    
    <div class="info-box">
        <strong>[TEST] Parameters:</strong><br>
        Strike Price (K): <span class="highlight">100.0</span> | 
        Maturity (T): <span class="highlight">1.0 year</span> | 
        Interest Rate (r): <span class="highlight">5%</span> | 
        Volatility (sigma): <span class="highlight">20%</span>
    </div>
    
    <div class="results">
        <h2>[RESULTS] Performance Metrics</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>MAE (Mean Absolute Error)</th>
                <th>RMSE</th>
                <th>Training Time</th>
                <th>Prediction Time</th>
            </tr>
            <tr>
                <td><strong>PINN</strong></td>
                <td><span class="highlight">26.5079</span></td>
                <td>36.8405</td>
                <td>0.89 s</td>
                <td>0.1252 ms</td>
            </tr>
            <tr>
                <td><strong>MLP</strong></td>
                <td><span class="highlight">18.6151</span></td>
                <td>20.5652</td>
                <td>0.09 s</td>
                <td>0.1070 ms</td>
            </tr>
        </table>
    </div>
    
    <div class="results">
        <h2>[FEATURES] PINN Advantages</h2>
        <ul>
            <li><strong>Theoretical Consistency:</strong> Integrates Black-Scholes PDE directly into learning process</li>
            <li><strong>Robustness to Data Scarcity:</strong> Maintains theoretical bounds even with limited data</li>
            <li><strong>Out-of-Domain Predictions:</strong> Better generalization beyond training data distribution</li>
            <li><strong>Physics-Guided Learning:</strong> Learns both data patterns and mathematical relationships</li>
        </ul>
    </div>
    
    <div class="graph-container">
        <h2>[GRAPHS] Comparison Visualizations</h2>
        <img src="/model_comparison_final.png" alt="PINN vs MLP Comparison">
        <p><em>Graph Descriptions:</em></p>
        <ul style="text-align: left; display: inline-block;">
            <li><strong>1. PINN Performance:</strong> True values vs PINN predictions</li>
            <li><strong>2. MLP Performance:</strong> True values vs MLP predictions</li>
            <li><strong>3. PINN vs MLP:</strong> Direct comparison of model predictions</li>
            <li><strong>4. PINN Error Distribution:</strong> Histogram of PINN prediction errors</li>
            <li><strong>5. MLP Error Distribution:</strong> Histogram of MLP prediction errors</li>
            <li><strong>6. MAE Comparison:</strong> Quick comparison of MAE metrics</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Generated: February 19, 2026</p>
        <p><strong>Graphics File:</strong> model_comparison_final.png</p>
        <p><strong>Location:</strong> /home/mehmetcelik/Masaüstü/masaüstü/makale/finance ml</p>
    </div>
</body>
</html>
"""
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
            
        elif self.path.startswith('/model_comparison_final.png'):
            # Serve PNG file
            try:
                with open(GRAPHICS_PATH, 'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.end_headers()
                    self.wfile.write(f.read())
            except Exception as e:
                self.send_response(404)
                self.end_headers()
                error_msg = f"File not found: {e}"
                self.wfile.write(error_msg.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Page not found")
    
    def log_message(self, format, *args):
        # Suppress log messages
        pass

if __name__ == '__main__':
    # Check if graphics file exists
    if not os.path.exists(GRAPHICS_PATH):
        print(f"[ERROR] Graphics file not found: {GRAPHICS_PATH}")
        exit(1)
    
    # Create HTTP server
    Handler = MyHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    
    print(f"[START] Graphics server starting...")
    print(f"[INFO] Open browser at: http://localhost:{PORT}")
    print(f"[INFO] Graphics file: {GRAPHICS_PATH}")
    print(f"[INFO] Press Ctrl+C to stop")
    
    # Open browser
    try:
        time.sleep(1)
        webbrowser.open(f'http://localhost:{PORT}')
    except Exception as e:
        print(f"[WARN] Could not auto-open browser: {e}")
    
    try:
        print(f"[RUNNING] Server listening on port {PORT}...")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[STOPPED] Server stopped")
        httpd.server_close()
        exit(0)
