CREATE TABLE IF NOT EXISTS assessments (
    id_assessment INTEGER PRIMARY KEY,
    assessment_type VARCHAR(50) NOT NULL,
    date SMALLINT NOT NULL,
    weight SMALLINT NOT NULL,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    assessment_type_ordinal SMALLINT NOT NULL,
    code_module_ordinal SMALLINT NOT NULL,
    code_presentation_ordinal SMALLINT NOT NULL,
    FOREIGN KEY (code_module, code_presentation) REFERENCES courses(code_module, code_presentation)
);

CREATE INDEX IF NOT EXISTS idx_assessments_module ON assessments(code_module);
CREATE INDEX IF NOT EXISTS idx_assessments_presentation ON assessments(code_presentation);