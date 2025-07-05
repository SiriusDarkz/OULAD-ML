CREATE TABLE IF NOT EXISTS courses_raw (
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    module_presentation_length SMALLINT,
    PRIMARY KEY (code_module, code_presentation)
);