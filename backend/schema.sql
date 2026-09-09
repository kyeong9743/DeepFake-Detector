-- Backend 실행시 자동으로 테이블을 생성하며, 이 파일을 참고용 MariaDB 스키마 파일
CREATE DATABASE IF NOT EXISTS cake CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'cake_app'@'%' IDENTIFIED BY '<비밀번호>';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX ON cake.* TO 'cake_app'@'%';
FLUSH PRIVILEGES;

USE cake;

CREATE TABLE analyses (
	id VARCHAR(32) NOT NULL, 
	created_at DATETIME NOT NULL, 
	updated_at DATETIME NOT NULL, 
	source_type VARCHAR(16) NOT NULL, 
	source_name VARCHAR(255) NOT NULL, 
	source_url TEXT, 
	media_path VARCHAR(255), 
	file_size INTEGER, 
	mode VARCHAR(8) NOT NULL, 
	client_ip VARCHAR(64), 
	status VARCHAR(16) NOT NULL, 
	stage VARCHAR(32) NOT NULL, 
	progress INTEGER NOT NULL, 
	message VARCHAR(255) NOT NULL, 
	error TEXT, 
	deepfake_score FLOAT, 
	confidence FLOAT, 
	risk_level VARCHAR(16), 
	is_deepfake BOOL, 
	cnn_score FLOAT, 
	lstm_score FLOAT, 
	frequency_score FLOAT, 
	vote_fake INTEGER, 
	frame_count INTEGER, 
	face_ratio FLOAT, 
	processing_time FLOAT, 
	video_duration FLOAT, 
	result_json JSON, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_analyses_created_at ON analyses (created_at);
CREATE INDEX ix_analyses_status ON analyses (status);
