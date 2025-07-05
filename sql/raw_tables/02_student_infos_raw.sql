CREATE TABLE IF NOT EXISTS student_infos_raw (
    id_student INT,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    gender VARCHAR(3),
    region VARCHAR(45),
    highest_education VARCHAR(45),
    imd_band VARCHAR(16),
    age_band VARCHAR(16),
    num_of_prev_attempts INT,
    studied_credits INT,
    disability VARCHAR(3),
    final_result VARCHAR(45),
    PRIMARY KEY (id_student, code_module, code_presentation),
    FOREIGN KEY (code_module, code_presentation)
        REFERENCES courses_raw(code_module, code_presentation)
);