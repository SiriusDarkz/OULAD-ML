CREATE OR REPLACE VIEW fd_student_vles AS
SELECT

	sv.id,
    sv.code_module,
    sv.code_presentation,
    sv.id_student,
    sv.id_site,
    sv.date,
    sv.sum_click,
    
    -- Campos de VLE
    v.activity_type,
    v.week_from,
    v.week_to,
    v.duration_weeks,

    -- Campos de student_info
    si.gender,
    si.gender_ordinal,
    si.region,
    si.region_ordinal,
    si.highest_education,
    si.highest_education_ordinal,
    si.imd_band,
    si.imd_band_ordinal,
    si.age_band,
    si.age_band_ordinal,
    si.num_of_prev_attempts,
    si.studied_credits,
    si.disability,
    si.disability_ordinal,
    si.final_result,
    si.final_result_ordinal,
    
    -- Campos de student_registration
    sr.date_registration,
    sr.date_unregistration,
    sr.was_dropped_out,
    sr.dropout_early,
    sr.registered_late,
    -- Feature engineering específico de VLE
    sv.activity_diversity,
    sv.early_engagement

FROM student_vles sv
LEFT JOIN vles v 
    ON sv.id_site = v.id_site
    AND sv.code_module = v.code_module
    AND sv.code_presentation = v.code_presentation

LEFT JOIN student_infos si 
    ON sv.id_student = si.id_student
    AND sv.code_module = si.code_module
    AND sv.code_presentation = si.code_presentation

LEFT JOIN student_registrations sr 
    ON sv.id_student = sr.id_student
    AND sv.code_module = sr.code_module
    AND sv.code_presentation = sr.code_presentation;