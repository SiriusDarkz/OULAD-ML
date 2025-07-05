CREATE TABLE IF NOT EXISTS student_assessments_raw (
    id_student INT,
    id_assessment INT,
    date_submitted INT,
    is_banked BOOLEAN,
    score FLOAT,
    PRIMARY KEY (id_student, id_assessment),
    FOREIGN KEY (id_assessment)
        REFERENCES assessments_raw(id_assessment)
);