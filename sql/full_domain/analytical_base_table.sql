/****************************************************************************************
 * SCRIPT PARA LA CREACIÓN DE LA TABLA ANALÍTICA BASE (ABT) FINAL
 * -----------------------------------------------------------------
 *
 * Propósito:
 * Crear una única tabla desnormalizada y numérica, lista para el análisis
 * exploratorio (EDA) y para entrenar modelos de Machine Learning.
 *
 * Características:
 * - 1 Fila = 1 Estudiante por Curso.
 * - Utiliza las columnas "_ordinal" pre-calculadas.
 * - Renombra todas las columnas a nombres más significativos y en inglés.
 * - Incluye características demográficas, de rendimiento y de comportamiento (engagement).
 *
 ****************************************************************************************/

-- Borrar la tabla si existe para asegurar una nueva creación limpia
DROP TABLE IF EXISTS analytical_base_table;

-- Crear la ABT usando el resultado de la consulta principal
CREATE TABLE analytical_base_table AS (

WITH
-- CTE 1: Agregamos el rendimiento académico de los estudiantes en las tareas
assessment_agg AS (
    SELECT
        sa.id_student,
        a.code_module_ordinal,
        a.code_presentation_ordinal,
        COUNT(sa.id_assessment) AS assignments_submitted_count,
        AVG(sa.score) AS assignments_avg_score,
        STDDEV(sa.score) AS assignments_score_std_dev,
        SUM(CASE WHEN sa.date_submitted > a.date THEN 1 ELSE 0 END) AS late_submissions_count
    FROM student_assessments sa
    JOIN assessments a 
        ON sa.id_assessment = a.id_assessment
    WHERE a.assessment_type != 'Exam'
    GROUP BY sa.id_student, a.code_module_ordinal, a.code_presentation_ordinal
),

-- CTE 2: Agregamos el comportamiento (engagement) de los estudiantes en el VLE
vle_agg AS (
    SELECT
        sv.id_student,
        sv.code_module_ordinal,
        sv.code_presentation_ordinal,
        SUM(sv.sum_click) AS total_clicks,
        COUNT(DISTINCT sv.date) AS days_engaged_count,
        SUM(CASE WHEN v.activity_type = 'oucontent' THEN sv.sum_click ELSE 0 END) AS clicks_content,
        SUM(CASE WHEN v.activity_type = 'forumng' THEN sv.sum_click ELSE 0 END) AS clicks_forum,
        SUM(CASE WHEN v.activity_type = 'quiz' THEN sv.sum_click ELSE 0 END) AS clicks_quiz,
        SUM(CASE WHEN v.activity_type = 'resource' THEN sv.sum_click ELSE 0 END) AS clicks_resource,
        SUM(CASE WHEN v.activity_type = 'subpage' THEN sv.sum_click ELSE 0 END) AS clicks_subpage,
        SUM(CASE WHEN v.activity_type = 'homepage' THEN sv.sum_click ELSE 0 END) AS clicks_homepage,
        SUM(CASE WHEN v.activity_type = 'questionnaire' THEN sv.sum_click ELSE 0 END) AS clicks_questionnaire,
        SUM(CASE WHEN v.activity_type = 'ouwiki' THEN sv.sum_click ELSE 0 END) AS clicks_ouwiki,
        SUM(CASE WHEN v.activity_type = 'htmlactivity' THEN sv.sum_click ELSE 0 END) AS clicks_htmlactivity,
        SUM(CASE WHEN v.activity_type = 'ouelluminate' THEN sv.sum_click ELSE 0 END) AS clicks_ouelluminate,
        SUM(CASE WHEN v.activity_type = 'dataplus' THEN sv.sum_click ELSE 0 END) AS clicks_dataplus,
        SUM(CASE WHEN v.activity_type = 'externalquiz' THEN sv.sum_click ELSE 0 END) AS clicks_externalquiz,
        SUM(CASE WHEN v.activity_type = 'repeatactivity' THEN sv.sum_click ELSE 0 END) AS clicks_repeatactivity,
        SUM(CASE WHEN v.activity_type = 'dualpane' THEN sv.sum_click ELSE 0 END) AS clicks_dualpane,
        SUM(CASE WHEN v.activity_type = 'glossary' THEN sv.sum_click ELSE 0 END) AS clicks_glossary,
        SUM(CASE WHEN v.activity_type = 'oucollaborate' THEN sv.sum_click ELSE 0 END) AS clicks_oucollaborate,
        SUM(CASE WHEN v.activity_type = 'folder' THEN sv.sum_click ELSE 0 END) AS clicks_folder
    FROM student_vles sv
    LEFT JOIN vles v ON sv.id_site = v.id_site
    GROUP BY sv.id_student, sv.code_module_ordinal, sv.code_presentation_ordinal
)

-- Consulta principal que une todo para formar la ABT
SELECT
    -- Identificadores principales
    si.id_student AS student_id,
    si.code_module_ordinal AS course_id,
    si.code_presentation_ordinal AS presentation_id,

    -- Características demográficas (ya numéricas)
    si.gender_ordinal AS gender,
    si.region_ordinal AS region,
    si.highest_education_ordinal AS education_level,
    si.imd_band_ordinal AS poverty_index_band,
    si.age_band_ordinal AS age_band,
    si.disability_ordinal AS has_disability,

    -- Características de experiencia previa
    si.num_of_prev_attempts AS previous_attempts_count,
    si.studied_credits AS studied_credits_total,

    -- Características de registro y actividad
    sr.date_registration AS registration_day,
    sr.date_unregistration AS unregistration_day,
    (CASE WHEN sr.date_unregistration IS NOT NULL THEN 1 ELSE 0 END) AS is_unregistered,
    (sr.date_unregistration - sr.date_registration) AS days_active_before_unreg,

    -- Características de rendimiento (de assessment_agg)
    COALESCE(aa.assignments_submitted_count, 0) AS assignments_submitted_count,
    COALESCE(aa.assignments_avg_score, 0) AS assignments_avg_score,
    COALESCE(aa.assignments_score_std_dev, 0) AS assignments_score_std_dev,
    COALESCE(aa.late_submissions_count, 0) AS late_submissions_count,

    -- Características de engagement (de vle_agg)
    COALESCE(va.total_clicks, 0) AS total_clicks,
    COALESCE(va.days_engaged_count, 0) AS days_engaged_count,
    COALESCE(va.clicks_content, 0) AS clicks_content,
    COALESCE(va.clicks_forum, 0) AS clicks_forum,
    COALESCE(va.clicks_quiz, 0) AS clicks_quiz,
    COALESCE(va.clicks_resource, 0) AS clicks_resource,
    COALESCE(va.clicks_subpage, 0) AS clicks_subpage,
    COALESCE(va.clicks_homepage, 0) AS clicks_homepage,
    COALESCE(va.clicks_questionnaire, 0) AS clicks_questionnaire,
    COALESCE(va.clicks_ouwiki, 0) AS clicks_ouwiki,
    COALESCE(va.clicks_htmlactivity, 0) AS clicks_htmlactivity,
    COALESCE(va.clicks_ouelluminate, 0) AS clicks_ouelluminate,
    COALESCE(va.clicks_dataplus, 0) AS clicks_dataplus,
    COALESCE(va.clicks_externalquiz, 0) AS clicks_externalquiz,
    COALESCE(va.clicks_repeatactivity, 0) AS clicks_repeatactivity,
    COALESCE(va.clicks_dualpane, 0) AS clicks_dualpane,
    COALESCE(va.clicks_glossary, 0) AS clicks_glossary,
    COALESCE(va.clicks_oucollaborate, 0) AS clicks_oucollaborate,
    COALESCE(va.clicks_folder, 0) AS clicks_folder,
    
    -- Variable Objetivo (Target)
    si.final_result_ordinal AS final_result

FROM student_infos si

LEFT JOIN student_registrations sr
    ON si.id_student = sr.id_student
    AND si.code_module_ordinal = sr.code_module_ordinal
    AND si.code_presentation_ordinal = sr.code_presentation_ordinal

LEFT JOIN assessment_agg aa
    ON si.id_student = aa.id_student
    AND si.code_module_ordinal = aa.code_module_ordinal
    AND si.code_presentation_ordinal = aa.code_presentation_ordinal

LEFT JOIN vle_agg va
    ON si.id_student = va.id_student
    AND si.code_module_ordinal = va.code_module_ordinal
    AND si.code_presentation_ordinal = va.code_presentation_ordinal
);