CREATE TABLE IF NOT EXISTS courses (
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    module_presentation_length SMALLINT NOT NULL,
    code_module_ordinal SMALLINT NOT NULL,
    code_presentation_ordinal SMALLINT NOT NULL,
    PRIMARY KEY (code_module, code_presentation)
);

CREATE INDEX IF NOT EXISTS idx_courses_module ON courses(code_module);
CREATE INDEX IF NOT EXISTS idx_courses_presentation ON courses(code_presentation);