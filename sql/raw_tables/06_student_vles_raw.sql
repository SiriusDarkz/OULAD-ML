CREATE TABLE IF NOT EXISTS student_vles_raw (
    id SERIAL PRIMARY KEY,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    id_student INT,
    id_site INT,
    date INT,
    sum_click INT,
    CONSTRAINT fk_student FOREIGN KEY (id_student, code_module, code_presentation)
        REFERENCES student_infos_raw(id_student, code_module, code_presentation),
    CONSTRAINT fk_vle FOREIGN KEY (id_site)
        REFERENCES vles_raw(id_site)
);