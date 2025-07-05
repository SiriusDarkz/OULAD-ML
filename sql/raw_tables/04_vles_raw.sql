CREATE TABLE IF NOT EXISTS vles_raw (
    id_site INT PRIMARY KEY,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    activity_type VARCHAR(45),
    week_from INT,
    week_to INT,
    FOREIGN KEY (code_module, code_presentation)
        REFERENCES courses_raw(code_module, code_presentation)
);