-- Create the database
CREATE DATABASE IF NOT EXISTS github_issues_db
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

-- Switch to the database
USE github_issues_db;

-- Create the issue table storing both labelled issues and unlabelled issues
CREATE TABLE IF NOT EXISTS issue (
    issue_id BIGINT PRIMARY KEY,    -- use BIGINT in case GitHub IDs are large
    content TEXT DEFAULT NULL,      -- full issue description
    summary TEXT DEFAULT NULL,                -- short summary (optional)
    type VARCHAR(128) DEFAULT NULL,    -- e.g. bug, feature, question
    priority VARCHAR(128) DEFAULT NULL -- e.g. high, medium, low
);
