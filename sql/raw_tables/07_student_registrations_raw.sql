CREATE TABLE IF NOT EXISTS student_registrations_raw (
    id_student INT,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    date_registration INT,
    date_unregistration INT,
    PRIMARY KEY (id_student, code_module, code_presentation),
    FOREIGN KEY (id_student, code_module, code_presentation)
        REFERENCES student_infos_raw(id_student, code_module, code_presentation)
);