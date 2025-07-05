CREATE TABLE IF NOT EXISTS student_vles (
    id SERIAL PRIMARY KEY,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    id_student INTEGER NOT NULL,
    id_site INTEGER NOT NULL,
    date SMALLINT NOT NULL,
    sum_click SMALLINT NOT NULL,
    code_module_ordinal SMALLINT NOT NULL,
    code_presentation_ordinal SMALLINT NOT NULL,
    FOREIGN KEY (code_module, code_presentation)
        REFERENCES courses(code_module, code_presentation),

    FOREIGN KEY (id_student, code_module, code_presentation)
        REFERENCES student_infos(id_student, code_module, code_presentation)
);

-- Índices para acelerar búsquedas comunes
CREATE INDEX IF NOT EXISTS idx_vle_student ON student_vles(id_student);
CREATE INDEX IF NOT EXISTS idx_vle_module ON student_vles(code_module);
CREATE INDEX IF NOT EXISTS idx_vle_site ON student_vles(id_site);