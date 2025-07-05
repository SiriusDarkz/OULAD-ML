import json
import os

class OrdinalEncoder:
    def __init__(self, mapping_dir):
        self.mapping_dir = mapping_dir
        self.mappings = {}

    def load_mapping(self, column_name):
        path = os.path.join(self.mapping_dir, f"{column_name}_map.json")
        with open(path, "r") as f:
            self.mappings[column_name] = json.load(f)

    def apply_mapping(self, df, column_name):
        if column_name not in self.mappings:
            self.load_mapping(column_name)
        mapping = self.mappings[column_name]
        new_col = f"{column_name}_ordinal"
        if df[column_name].dtype == "bool":
            df[column_name] = df[column_name].astype(str)
        df[new_col] = df[column_name].map(mapping)
        return df

    def apply_all(self, df, columns):
        for col in columns:
            df = self.apply_mapping(df, col)
        return df

# Función externa para uso directo en el pipeline

def apply_all_encodings(df, table_name):
    encoder = OrdinalEncoder(mapping_dir="mappings")

    columns_by_table = {
        "student_infos": [
            "gender", "region", "highest_education", "imd_band",
            "age_band", "disability", "final_result",
            "code_module", "code_presentation"
        ],
        "student_registrations": [
            "code_module", 
            "code_presentation"
        ],
        "student_assessments": [
            "is_banked"
        ],
        "assessments": [
            "assessment_type", 
            "code_module",
            "code_presentation"
        ],
        "vles": [
            "activity_type", 
            "code_module", 
            "code_presentation"
        ],
        "student_vles": [
            "code_module",
            "code_presentation"
        ],
        "courses": [
            "code_module", 
            "code_presentation"
        ]
    }

    columns = columns_by_table.get(table_name, [])
    return encoder.apply_all(df, columns)
