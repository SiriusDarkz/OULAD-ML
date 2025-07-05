CREATE TEMP TABLE estudiantes_a_eliminar AS
SELECT DISTINCT id_student
FROM student_registrations_raw
WHERE date_registration IS NULL;

SELECT count(*) FROM estudiantes_a_eliminar;

DELETE FROM student_assessments_raw
WHERE id_student IN (SELECT id_student FROM estudiantes_a_eliminar);

DELETE FROM student_vles_raw
WHERE id_student IN (SELECT id_student FROM estudiantes_a_eliminar);

DELETE FROM student_registrations_raw
WHERE id_student IN (SELECT id_student FROM estudiantes_a_eliminar);

DELETE FROM student_infos_raw
WHERE id_student IN (SELECT id_student FROM estudiantes_a_eliminar);
