CREATE MATERIALIZED VIEW IF NOT EXISTS fd_student_assessments AS
SELECT
    -- Identificadores clave
    sa.id_assessment,
    sa.id_student,
    a.code_module,
    a.code_presentation,

    -- Datos del assessment
    sa.date_submitted,
    sa.is_banked,
    sa.score,
    a.date AS assessment_date,
    a.weight,
    a.assessment_type,

    -- Datos de registro del estudiante
    sr.date_registration,
    sr.date_unregistration,

    -- Datos de demographics (de student_info)
    si.region,
    si.highest_education,
    si.imd_band,
    si.age_band,
    si.num_of_prev_attempts,
    si.studied_credits,
    si.disability,
    si.final_result,

    -- Ordinales útiles
    si.gender_ordinal,
    si.region_ordinal,
    si.highest_education_ordinal,
    si.imd_band_ordinal,
    si.age_band_ordinal,
    si.disability_ordinal,
    si.final_result_ordinal,
    si.code_module_ordinal,
    si.code_presentation_ordinal,

    -- Score ponderado
ROUND((sa.score * a.weight / 100.0)::numeric, 2) AS score_weighted,

    -- VLE: días distintos por tipo de actividad
    COUNT(DISTINCT CASE WHEN v.activity_type = 'dataplus' THEN sv.date ELSE NULL END) AS n_days_dataplus,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'dualpane' THEN sv.date ELSE NULL END) AS n_days_dualpane,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'externalquiz' THEN sv.date ELSE NULL END) AS n_days_externalquiz,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'folder' THEN sv.date ELSE NULL END) AS n_days_folder,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'forumng' THEN sv.date ELSE NULL END) AS n_days_forumng,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'glossary' THEN sv.date ELSE NULL END) AS n_days_glossary,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'homepage' THEN sv.date ELSE NULL END) AS n_days_homepage,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'htmlactivity' THEN sv.date ELSE NULL END) AS n_days_htmlactivity,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'imscp' THEN sv.date ELSE NULL END) AS n_days_imscp,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'other' THEN sv.date ELSE NULL END) AS n_days_other,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'questionnaire' THEN sv.date ELSE NULL END) AS n_days_questionnaire,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'repeatactivity' THEN sv.date ELSE NULL END) AS n_days_repeatactivity,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'resource' THEN sv.date ELSE NULL END) AS n_days_resource,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'sharedsubpage' THEN sv.date ELSE NULL END) AS n_days_sharedsubpage,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'subpage' THEN sv.date ELSE NULL END) AS n_days_subpage,
    COUNT(DISTINCT CASE WHEN v.activity_type = 'url' THEN sv.date ELSE NULL END) AS n_days_url,

    -- Total días con actividad
    COUNT(DISTINCT sv.date) AS total_days_vle

FROM
    student_assessments sa
JOIN
    assessments a ON sa.id_assessment = a.id_assessment
LEFT JOIN
    student_registrations sr ON sa.id_student = sr.id_student 
        AND a.code_module = sr.code_module 
        AND a.code_presentation = sr.code_presentation
LEFT JOIN
    student_infos si ON sa.id_student = si.id_student 
        AND a.code_module = si.code_module 
        AND a.code_presentation = si.code_presentation
LEFT JOIN
    student_vles sv ON sa.id_student = sv.id_student 
        AND a.code_module = sv.code_module 
        AND a.code_presentation = sv.code_presentation
LEFT JOIN
    vles v ON sv.id_site = v.id_site 
        AND a.code_module = v.code_module 
        AND a.code_presentation = v.code_presentation

GROUP BY
    sa.id_assessment,
    sa.id_student,
    a.code_module,
    a.code_presentation,
    sa.date_submitted,
    sa.is_banked,
    sa.score,
    a.date,
    a.weight,
    a.assessment_type,
    sr.date_registration,
    sr.date_unregistration,
    si.region,
    si.highest_education,
    si.imd_band,
    si.age_band,
    si.num_of_prev_attempts,
    si.studied_credits,
    si.disability,
    si.final_result,
    si.gender_ordinal,
    si.region_ordinal,
    si.highest_education_ordinal,
    si.imd_band_ordinal,
    si.age_band_ordinal,
    si.disability_ordinal,
    si.final_result_ordinal,
    si.code_module_ordinal,
    si.code_presentation_ordinal;