CREATE TABLE IF NOT EXISTS student_registrations (
    id_student INTEGER NOT NULL,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    date_registration SMALLINT NOT NULL,
    date_unregistration SMALLINT,
    code_module_ordinal SMALLINT NOT NULL,
    code_presentation_ordinal SMALLINT NOT NULL,
    PRIMARY KEY (id_student, code_module, code_presentation),
    FOREIGN KEY (id_student, code_module, code_presentation)
        REFERENCES student_infos(id_student, code_module, code_presentation)
);

CREATE INDEX IF NOT EXISTS idx_studentreg_student ON student_registrations(id_student);
CREATE INDEX IF NOT EXISTS idx_studentreg_module ON student_registrations(code_module);