CREATE TABLE IF NOT EXISTS student_infos (
    id_student INTEGER NOT NULL,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    gender VARCHAR(10) NOT NULL,
    region VARCHAR(100) NOT NULL,
    highest_education VARCHAR(100) NOT NULL,
    imd_band VARCHAR(20) NOT NULL,
    age_band VARCHAR(20) NOT NULL,
    disability VARCHAR(10) NOT NULL,
    final_result VARCHAR(20) NOT NULL,
    gender_ordinal SMALLINT NOT NULL,
    region_ordinal SMALLINT NOT NULL,
    highest_education_ordinal SMALLINT NOT NULL,
    imd_band_ordinal SMALLINT NOT NULL,
    age_band_ordinal SMALLINT NOT NULL,
    disability_ordinal SMALLINT NOT NULL,
    num_of_prev_attempts INT NOT NULL,
    studied_credits INT NOT NULL,
    final_result_ordinal SMALLINT NOT NULL,
    code_module_ordinal SMALLINT NOT NULL,
    code_presentation_ordinal SMALLINT NOT NULL,
    PRIMARY KEY (id_student, code_module, code_presentation),
    FOREIGN KEY (code_module, code_presentation) REFERENCES courses(code_module, code_presentation)
);

CREATE INDEX IF NOT EXISTS idx_studentinfo_student ON student_infos(id_student);
CREATE INDEX IF NOT EXISTS idx_studentinfo_module ON student_infos(code_module);
CREATE INDEX IF NOT EXISTS idx_studentinfo_presentation ON student_infos(code_presentation);