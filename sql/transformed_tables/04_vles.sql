CREATE TABLE IF NOT EXISTS vles (
    id_site INTEGER PRIMARY KEY,
    code_module VARCHAR(45) NOT NULL,
    code_presentation VARCHAR(45) NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    week_from SMALLINT NOT NULL,
    week_to SMALLINT NOT NULL,
    activity_type_ordinal SMALLINT NOT NULL,
    code_module_ordinal SMALLINT NOT NULL,
    code_presentation_ordinal SMALLINT NOT NULL,
    FOREIGN KEY (code_module, code_presentation) REFERENCES courses(code_module, code_presentation)
);

CREATE INDEX IF NOT EXISTS idx_vle_module ON vles(code_module);
CREATE INDEX IF NOT EXISTS idx_vle_activity_type ON vles(activity_type);