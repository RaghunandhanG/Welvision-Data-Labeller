-- Create WelVision Database and Models Table
-- Execute this script to set up the database for YOLO model management

-- Create database
CREATE DATABASE IF NOT EXISTS welvision_db;
USE welvision_db;

-- Create ai_models table
CREATE TABLE IF NOT EXISTS ai_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(500) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(500) NOT NULL,
    description TEXT,
    image_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Insert sample model data
INSERT INTO ai_models (name, path, description) VALUES 
('Head Test Aug 12', 'C:\\Users\\raghu\\AppData\\Local\\Microsoft\\Windows\\INetCache\\IE\\FARWN71Z\\runs[1].zip\\runs\\detect\\Head Test Aug 12\\weights\\best.pt', 'Head detection model trained in August 2012'),
('Default YOLO', 'yolov8n.pt', 'Default YOLOv8 nano model'),
('Custom Object Detection', 'models/custom_od.pt', 'Custom object detection model');

-- Show created table structures
DESCRIBE ai_models;
DESCRIBE datasets;

-- Show inserted data
SELECT * FROM ai_models;
SELECT * FROM datasets; 