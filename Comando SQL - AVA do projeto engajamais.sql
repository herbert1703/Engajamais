===========================================
/* Scripts_SQL_Projeto_ead_script_Inicial */
===========================================
-- Criar a view 'alunos'
CREATE OR REPLACE VIEW alunos AS
SELECT c.id AS "disciplina_id", u.id AS "aluno_id"
FROM mdl_role_assignments rs
INNER JOIN mdl_context e ON rs.contextid=e.id
INNER JOIN mdl_course c ON c.id = e.instanceid
INNER JOIN mdl_user u ON u.id=rs.userid
WHERE e.contextlevel=50 AND rs.roleid=5
ORDER BY c.id, u.id;
===========================================
-- Criar a view 'professores'
CREATE OR REPLACE VIEW professores AS
SELECT c.id AS "disciplina_id", u.id AS "professor_id"
FROM mdl_role_assignments rs
INNER JOIN mdl_context e ON rs.contextid=e.id
INNER JOIN mdl_course c ON c.id = e.instanceid
INNER JOIN mdl_user u ON u.id=rs.userid
WHERE e.contextlevel=50 AND rs.roleid IN (2,3,4,9,10,12,13)
ORDER BY c.id, u.id;
===========================================
-- Criar view 'id_alunos' => base = alunos
CREATE OR REPLACE VIEW id_alunos AS
SELECT distinct(aluno_id) FROM alunos
ORDER BY aluno_id;
==========================================
-- Criar view 'id_disciplinas' => base = alunos
CREATE OR REPLACE VIEW id_disciplinas AS
SELECT distinct(disciplina_id) FROM alunos
ORDER BY disciplina_id;
==========================================
-- Criar view 'log_reduzido'
CREATE OR REPLACE VIEW log_reduzido AS
SELECT ( row_number() OVER(ORDER BY userid,timecreated)) AS id, timecreated,
userid, courseid, component, action
FROM
(SELECT * FROM mdl_logstore_standard_log WHERE courseid IN (SELECT * FROM
id_disciplinas) AND userid IN (SELECT * FROM id_alunos)
UNION
SELECT * FROM mdl_logstore_standard_log WHERE action ='loggedout' AND
userid IN (SELECT * FROM id_alunos)
UNION
SELECT * FROM mdl_logstore_standard_log WHERE action='loggedin' AND
userid IN (SELECT * FROM id_alunos)) log;
===========================================
/*  TABELA BASE */
CREATE TABLE `ead`.`base` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `codcurso` VARCHAR(5) NULL,
  `curso` VARCHAR(160) NULL,
  `semestre` VARCHAR(6) NULL,
  `periodo` INT NULL,
  `disciplina_nome` VARCHAR(160) NULL,
  `idnumber` VARCHAR(10) NULL,
  `disciplina_id` VARCHAR(10) NULL,
  `data_inicio` DATETIME NULL,
  `data_final` DATETIME NULL,
  `username` VARCHAR(11) NULL,
  `aluno_nome` VARCHAR(30) NULL,
  `aluno_id` VARCHAR(10) NULL,
  `ra` VARCHAR(10) NULL,
  `sexo` VARCHAR(1) NULL,
  `email` VARCHAR(60) NULL,
  `uf` VARCHAR(2) NULL,
  `cidade` VARCHAR(60) NULL,
  `bairro` VARCHAR(70) NULL,
  `status` VARCHAR(45) NULL,
  `nascimento` DATETIME NULL,
  PRIMARY KEY (`id`));

==========================================

update base set disciplina_id = (select id from moodle.mdl_course where moodle.mdl_course.idnumber = base.idnumber)
where id >= 1;

update base set aluno_id = (select id from moodle.mdl_user where moodle.mdl_user.username = base.username)
where id >= 1;

===========================================

/* Comando para o caso do Mysql 8.0.30 */
SET GLOBAL sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));

===========================================

/*Criação de índices para melhorar o tempo de busca das informações*/
create index idx_contextview on mdl_context (contextlevel);
create index idx_base on base (curso, semestre,periodo, disciplina_id,aluno_id,aluno_nome);
create index idx_comp_log on mdl_logstore_standard_log (contextinstanceid,component,action);
create index idx_semestre on base (semestre);
create index idx_mdllogred on mdl_logstore_standard_log(userid,timecreated,courseid);
create index idx_mdlmsn_read on mdl_message_read(useridfrom,timecreated);


===========================================
/* Selects para busca dos dados */
===========================================
/* Select completo com algumas mudanças para o caso de utilização com integração de CMS */

SELECT 
base.curso AS "Curso",
base.semestre AS "Semestre",
base.periodo AS "Período",
base.disciplina_nome AS "Nome da Disciplina",
base.disciplina_id AS "ID da Disciplina",
base.data_inicio AS "Data de Início",
base.data_final AS "Data de Final",
base.aluno_nome AS "Nome do Aluno",
base.aluno_id AS "ID do Aluno",
COALESCE(var01,0) AS "var01",
--	COALESCE(var02,0) AS "var02",
--	COALESCE(var03,0) AS "var03",
COALESCE(var04,0) AS "var04",
COALESCE(var05,0) AS "var05",
--	COALESCE(var06,0) AS "var06",
COALESCE(var07,0) AS "var07",
COALESCE(var08,0) AS "var08",
COALESCE(var09,0) AS "var09",
--      COALESCE(var10,0) AS "var10",
--	COALESCE(var11,0) AS "var11",
--	COALESCE(var12,0) AS "var12",
COALESCE(var13a,0) AS "var13a",
COALESCE(var13b,0) AS "var13b",
COALESCE(var13c,0) AS "var13c",
COALESCE(var13d,0) AS "var13d",
COALESCE(var17,0) AS "var17",
COALESCE(var18,0) AS "var18",
COALESCE(var19,0) AS "var19",
COALESCE(var24,0) AS "var24",
COALESCE(var27,0) AS "var27",
COALESCE(var34,0) AS "var34"
-- Junções que determinam os valores das variáveis para cada aluno
FROM
(SELECT * FROM base) AS base
	LEFT OUTER JOIN (SELECT b.disciplina_id, b.aluno_id, count(*) AS "var01"
		FROM mdl_forum_posts p
		INNER JOIN mdl_forum_discussions d ON d.id = p.discussion
		INNER JOIN base b ON d.course = b.disciplina_id AND p.userid=b.aluno_id AND p.created BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id, b.aluno_id) AS VAR01
	ON VAR01.disciplina_id = base.disciplina_id AND VAR01.aluno_id = base.aluno_id

--		LEFT OUTER JOIN (SELECT p1.disciplina_id,receptor, count(*) AS "var02"
--			FROM mdl_post p1
--			INNER JOIN professores p2 ON p2.disciplina_id=p1.disciplina_id AND p2.professor_id=p1.emissor
--			INNER JOIN alunos p3 ON p3.disciplina_id=p1.disciplina_id AND p3.aluno_id=p1.receptor
--			INNER JOIN base b ON b.disciplina_id=p1.disciplina_id AND b.aluno_id=p1.receptor AND p1.created BETWEEN @data_inicio and @data_final
--			GROUP BY p1.disciplina_id, receptor) AS var02
--		ON var02.disciplina_id = base.disciplina_id AND var02.receptor = base.aluno_id
--		LEFT OUTER JOIN (SELECT p1.disciplina_id,receptor, count(*) AS "var03"
--			FROM mdl_post p1
--			INNER JOIN base b ON b.disciplina_id=p1.disciplina_id AND b.aluno_id=p1.receptor AND p1.created BETWEEN @data_inicio and @data_final
--			INNER JOIN alunos p2 ON p2.disciplina_id=p1.disciplina_id AND p2.aluno_id=p1.emissor
--			GROUP BY p1.disciplina_id, receptor) AS var03
--		ON var03.disciplina_id = base.disciplina_id AND var03.receptor = base.aluno_id
	LEFT OUTER JOIN (SELECT b.disciplina_id, b.aluno_id, count(*) AS "var04"
		FROM mdl_message_read r
		INNER JOIN base b ON b.aluno_id=r.useridfrom AND r.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id, b.aluno_id) AS var04
	ON var04.disciplina_id = base.disciplina_id AND var04.aluno_id = base.aluno_id
	LEFT OUTER JOIN (SELECT b.disciplina_id, b.aluno_id, count(*) AS "var05"
		FROM mdl_message_read r
		INNER JOIN base b ON b.aluno_id=r.useridto AND r.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id, b.aluno_id) AS var05
	ON var05.disciplina_id = base.disciplina_id AND var05.aluno_id = base.aluno_id
	LEFT OUTER JOIN (SELECT temp.disciplina_id, count(*) AS "var07"
		FROM (SELECT b.disciplina_id,component, count(*)
			FROM (SELECT distinct(disciplina_id), data_inicio, data_final FROM base) b
		INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE contextinstanceid > 0 AND (component = "mod_lti" OR component = "mod_forum" OR component = "mod_quiz")) l    /*Atividades*/ 
		ON b.disciplina_id=l.courseid AND l.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id,l.component) AS temp
	GROUP BY temp.disciplina_id) AS var07
	ON var07.disciplina_id = base.disciplina_id
	LEFT OUTER JOIN (SELECT temp.disciplina_id, count(*) AS "var08"
		FROM (SELECT b.disciplina_id,component,contextinstanceid, count(*)
	FROM (SELECT distinct(disciplina_id), data_inicio, data_final FROM base) b
	INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE contextinstanceid>0 AND (component = "mod_resource" OR component = "mod_folder" OR component = "mod_glossary")) l     /*Recursos*/
	ON b.disciplina_id=l.courseid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,l.component,contextinstanceid) AS temp
GROUP BY temp.disciplina_id) AS var08
ON var08.disciplina_id = base.disciplina_id

LEFT OUTER JOIN    
(SELECT tmp.disciplina_id,sum(tmp.totgeral) as "var09" from 
(SELECT temp.disciplina_id,temp.aluno_id, ROUND(temp.total) AS "totgeral"
FROM (SELECT b.disciplina_id,b.aluno_id, component,contextinstanceid, count(*) AS "total"
	FROM base b
	INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE contextinstanceid>0 AND
	/*Atividades*/
	(component = "mod_lti" AND action="viewed") ) l
	ON b.disciplina_id=l.courseid AND b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id,l.component,contextinstanceid) AS temp
    GROUP BY temp.disciplina_id, temp.aluno_id)  as tmp
    GROUP BY tmp.disciplina_id) as var09
    ON var09.disciplina_id = base.disciplina_id
    
-- LEFT OUTER JOIN (SELECT temp.disciplina_id,temp.aluno_id, ROUND(AVG(temp.total),2) AS "var10"
-- 	FROM (SELECT b.disciplina_id,b.aluno_id, component,contextinstanceid, count(*) AS "total"
-- 		FROM base b
--		INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE contextinstanceid>0 AND /*Atividades*/ (component = "mod_lti" AND -- action="view submission") OR
--			(component = "mod_forum" AND action="view forum") OR (component = "mod_quiz" AND action="view") OR
--			/*Recursos considerados*/ 
--			(component = "mod_resource" AND action="view") OR (component="mod_folder" AND action="view") OR (component = -- "mod_glossary" AND action="view")) l
--		ON b.disciplina_id=l.courseid AND b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
--		GROUP BY b.disciplina_id,b.aluno_id,l.component,contextinstanceid) AS temp
--	GROUP BY temp.disciplina_id, temp.aluno_id) AS var10
--	ON var10.disciplina_id = base.disciplina_id AND var10.aluno_id = base.aluno_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13a"
		FROM base b
		INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 6 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 12) l
		ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id,b.aluno_id) AS var13a
	ON var13a.aluno_id = base.aluno_id AND var13a.disciplina_id = base.disciplina_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13b"
		FROM base b
		INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 12 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 18) l
		ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id,b.aluno_id) AS var13b
	ON var13b.aluno_id = base.aluno_id AND var13b.disciplina_id = base.disciplina_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13c"
	FROM base b
	INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 18 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 24) l
	ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id) AS var13c
ON var13c.aluno_id = base.aluno_id AND var13c.disciplina_id = base.disciplina_id
LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13d"
FROM base b
INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 0 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 6) l
ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
GROUP BY b.disciplina_id,b.aluno_id) AS var13d
ON var13d.aluno_id = base.aluno_id AND var13d.disciplina_id = base.disciplina_id
LEFT OUTER JOIN (SELECT d.courseid,d.userid, SUM(d.intervalo) AS "var17" 
	FROM (SELECT c.courseid, c.userid, 
									 CASE WHEN c.proxima_action="loggedin" THEN 0
		WHEN c.proximo_time= NULL THEN 0
		ELSE c.proximo_time - c.timecreated END AS "intervalo"
			 FROM (SELECT a.id,a.courseid,a.userid,a.component,a.action,a.timecreated,b.action AS "proxima_action", b.timecreated AS "proximo_time" 
			FROM 
            (SELECT ( row_number() OVER(ORDER BY userid,timecreated)) AS id, timecreated,
userid, courseid, component, action 
FROM
(SELECT log.* FROM mdl_logstore_standard_log log WHERE (SELECT count(*) FROM
id_disciplinas i_dsc where i_dsc.disciplina_id = log.courseid) > 0 AND 
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action ='loggedout' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action='loggedin' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
) log) a 
INNER JOIN 
(SELECT ( row_number() OVER(ORDER BY userid,timecreated)) AS id, timecreated,
userid, courseid, component, action 
FROM
(SELECT log.* FROM mdl_logstore_standard_log log WHERE (SELECT count(*) FROM
id_disciplinas i_dsc where i_dsc.disciplina_id = log.courseid) > 0 AND 
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action ='loggedout' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action='loggedin' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
) log) b ON a.userid=b.userid AND (b.id-1)=a.id) c
	INNER JOIN (SELECT distinct(disciplina_id) 
	FROM base) b ON c.courseid=b.disciplina_id AND c.timecreated BETWEEN @data_inicio and @data_final
		) d
		GROUP BY d.courseid,d.userid) AS var17
	ON var17.courseid = base.disciplina_id AND var17.userid = base.aluno_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var18"
			FROM base b
			INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin") l
		ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id,b.aluno_id) AS var18
	ON var18.aluno_id = base.aluno_id AND var18.disciplina_id = base.disciplina_id    
	LEFT OUTER JOIN (SELECT b.disciplina_id, b.aluno_id, count(*) AS "var19"
	FROM mdl_message_read r
	INNER JOIN base b ON b.aluno_id=r.useridfrom AND r.timecreated BETWEEN @data_inicio and @data_final
	INNER JOIN professores p ON p.professor_id=r.useridto AND p.disciplina_id=b.disciplina_id
	GROUP BY b.disciplina_id, b.aluno_id) AS var19
    ON var19.disciplina_id = base.disciplina_id AND var19.aluno_id = base.aluno_id
    LEFT OUTER JOIN (SELECT temp.disciplina_id,temp.aluno_id, count(*) AS "var24"
		FROM (SELECT b.disciplina_id,b.aluno_id, ip, count(*) AS "Num_Acesso_IP"
	FROM (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin") l
	INNER JOIN base b ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id,l.ip) AS temp
GROUP BY temp.disciplina_id, temp.aluno_id) AS var24
ON var24.disciplina_id = base.disciplina_id AND var24.aluno_id = base.aluno_id
LEFT OUTER JOIN (SELECT temp.disciplina_id,temp.aluno_id, ROUND(temp.total) AS "var27"
FROM (SELECT b.disciplina_id,b.aluno_id, component,contextinstanceid, count(*) AS "total"
	FROM base b
	INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE contextinstanceid>0 AND
	/*Atividades*/
	(component = "mod_lti" AND action="viewed") ) l
	ON b.disciplina_id=l.courseid AND b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id,l.component,contextinstanceid) AS temp
GROUP BY temp.disciplina_id, temp.aluno_id) AS var27
ON var27.disciplina_id = base.disciplina_id AND var27.aluno_id = base.aluno_id
LEFT OUTER JOIN (SELECT b.disciplina_id, b.aluno_id, tmp.gradepercent AS "var34"
FROM (select re.course, re.userid, avg(re.gradepercent) as gradepercent from 
		(select lt.id, lt.course, sb.userid, max(sb.gradepercent) as gradepercent from mdl_lti lt
		   inner join mdl_lti_submission sb on sb.ltiid = lt.id
			group by lt.course, sb.userid, lt.id) re
		 group by re.course, re.userid) tmp 
INNER JOIN base b ON b.disciplina_id = tmp.course AND b.aluno_id = tmp.userid 
GROUP BY b.disciplina_id, b.aluno_id) AS var34
ON var34.disciplina_id = base.disciplina_id AND var34.aluno_id = base.aluno_id
where base.semestre = '2018.1'
ORDER BY
base.curso, base.semestre, base.periodo, base.disciplina_nome, base.aluno_nome
--	)

===========================================
/* Select reduzido apenas com as variáveis utilizadas no projeto e algumas mudanças para o caso de utilização com integração de CMS */

SELECT 
base.curso AS "Curso",
base.semestre AS "Semestre",
base.periodo AS "Período",
base.disciplina_nome AS "Nome da Disciplina",
base.disciplina_id AS "ID da Disciplina",
base.data_inicio AS "Data de Início",
base.data_final AS "Data de Final",
base.aluno_nome AS "Nome do Aluno",
base.aluno_id AS "ID do Aluno",
COALESCE(var01,0) AS "var01",
COALESCE(var09,0) AS "var09",
COALESCE(var13a,0) AS "var13a",
COALESCE(var13b,0) AS "var13b",
COALESCE(var13c,0) AS "var13c",
COALESCE(var13d,0) AS "var13d",
COALESCE(var17,0) AS "var17",
COALESCE(var18,0) AS "var18",
COALESCE(var24,0) AS "var24",
COALESCE(var27,0) AS "var27",
COALESCE(var34,0) AS "var34"
-- Junções que determinam os valores das variáveis para cada aluno
FROM
(SELECT * FROM base) AS base
	LEFT OUTER JOIN (SELECT b.disciplina_id, b.aluno_id, count(*) AS "var01"
		FROM mdl_forum_posts p
		INNER JOIN mdl_forum_discussions d ON d.id = p.discussion
		INNER JOIN base b ON d.course = b.disciplina_id AND p.userid=b.aluno_id AND p.created BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id, b.aluno_id) AS VAR01
	ON VAR01.disciplina_id = base.disciplina_id AND VAR01.aluno_id = base.aluno_id
LEFT OUTER JOIN    
(SELECT tmp.disciplina_id,sum(tmp.totgeral) as "var09" from 
(SELECT temp.disciplina_id,temp.aluno_id, ROUND(temp.total) AS "totgeral"
FROM (SELECT b.disciplina_id,b.aluno_id, component,contextinstanceid, count(*) AS "total"
	FROM base b
	INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE contextinstanceid>0 AND
	/*Atividades*/
	(component = "mod_lti" AND action="viewed") ) l
	ON b.disciplina_id=l.courseid AND b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id,l.component,contextinstanceid) AS temp
    GROUP BY temp.disciplina_id, temp.aluno_id)  as tmp
    GROUP BY tmp.disciplina_id) as var09
    ON var09.disciplina_id = base.disciplina_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13a"
		FROM base b
		INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 6 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 12) l
		ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id,b.aluno_id) AS var13a
	ON var13a.aluno_id = base.aluno_id AND var13a.disciplina_id = base.disciplina_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13b"
		FROM base b
		INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 12 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 18) l
		ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id,b.aluno_id) AS var13b
	ON var13b.aluno_id = base.aluno_id AND var13b.disciplina_id = base.disciplina_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13c"
	FROM base b
	INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 18 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 24) l
	ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id) AS var13c
ON var13c.aluno_id = base.aluno_id AND var13c.disciplina_id = base.disciplina_id
LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var13d"
FROM base b
INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin" AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) >= 0 AND EXTRACT(HOUR FROM FROM_UNIXTIME(timecreated)) < 6) l
ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
GROUP BY b.disciplina_id,b.aluno_id) AS var13d
ON var13d.aluno_id = base.aluno_id AND var13d.disciplina_id = base.disciplina_id
LEFT OUTER JOIN (SELECT d.courseid,d.userid, SUM(d.intervalo) AS "var17" 
	FROM (SELECT c.courseid, c.userid, 
									 CASE WHEN c.proxima_action="loggedin" THEN 0
		WHEN c.proximo_time= NULL THEN 0
		ELSE c.proximo_time - c.timecreated END AS "intervalo"
			 FROM (SELECT a.id,a.courseid,a.userid,a.component,a.action,a.timecreated,b.action AS "proxima_action", b.timecreated AS "proximo_time" 
			FROM 
            (SELECT ( row_number() OVER(ORDER BY userid,timecreated)) AS id, timecreated,
userid, courseid, component, action 
FROM
(SELECT log.* FROM mdl_logstore_standard_log log WHERE (SELECT count(*) FROM
id_disciplinas i_dsc where i_dsc.disciplina_id = log.courseid) > 0 AND 
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action ='loggedout' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action='loggedin' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
) log) a 
INNER JOIN 
(SELECT ( row_number() OVER(ORDER BY userid,timecreated)) AS id, timecreated,
userid, courseid, component, action 
FROM
(SELECT log.* FROM mdl_logstore_standard_log log WHERE (SELECT count(*) FROM
id_disciplinas i_dsc where i_dsc.disciplina_id = log.courseid) > 0 AND 
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action ='loggedout' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
UNION
SELECT log.* FROM mdl_logstore_standard_log log WHERE log.action='loggedin' AND
(SELECT count(*) FROM id_alunos ialuno where ialuno.aluno_id = log.userid) > 0
and log.timecreated BETWEEN @data_inicio and @data_final
) log) b ON a.userid=b.userid AND (b.id-1)=a.id) c
	INNER JOIN (SELECT distinct(disciplina_id) 
	FROM base) b ON c.courseid=b.disciplina_id AND c.timecreated BETWEEN @data_inicio and @data_final
		) d
		GROUP BY d.courseid,d.userid) AS var17
	ON var17.courseid = base.disciplina_id AND var17.userid = base.aluno_id
	LEFT OUTER JOIN (SELECT b.disciplina_id,b.aluno_id, count(*) AS "var18"
			FROM base b
			INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin") l
		ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
		GROUP BY b.disciplina_id,b.aluno_id) AS var18
	ON var18.aluno_id = base.aluno_id AND var18.disciplina_id = base.disciplina_id    
    LEFT OUTER JOIN (SELECT temp.disciplina_id,temp.aluno_id, count(*) AS "var24"
		FROM (SELECT b.disciplina_id,b.aluno_id, ip, count(*) AS "Num_Acesso_IP"
	FROM (SELECT * FROM mdl_logstore_standard_log WHERE action="loggedin") l
	INNER JOIN base b ON b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id,l.ip) AS temp
GROUP BY temp.disciplina_id, temp.aluno_id) AS var24
ON var24.disciplina_id = base.disciplina_id AND var24.aluno_id = base.aluno_id
LEFT OUTER JOIN (SELECT temp.disciplina_id,temp.aluno_id, ROUND(temp.total) AS "var27"
FROM (SELECT b.disciplina_id,b.aluno_id, component,contextinstanceid, count(*) AS "total"
	FROM base b
	INNER JOIN (SELECT * FROM mdl_logstore_standard_log WHERE contextinstanceid>0 AND
	/*Atividades*/
	(component = "mod_lti" AND action="viewed") ) l
	ON b.disciplina_id=l.courseid AND b.aluno_id=l.userid AND l.timecreated BETWEEN @data_inicio and @data_final
	GROUP BY b.disciplina_id,b.aluno_id,l.component,contextinstanceid) AS temp
GROUP BY temp.disciplina_id, temp.aluno_id) AS var27
ON var27.disciplina_id = base.disciplina_id AND var27.aluno_id = base.aluno_id
LEFT OUTER JOIN (SELECT b.disciplina_id, b.aluno_id, tmp.gradepercent AS "var34"
FROM (select re.course, re.userid, avg(re.gradepercent) as gradepercent from 
		(select lt.id, lt.course, sb.userid, max(sb.gradepercent) as gradepercent from mdl_lti lt
		   inner join mdl_lti_submission sb on sb.ltiid = lt.id
			group by lt.course, sb.userid, lt.id) re
		 group by re.course, re.userid) tmp 
INNER JOIN base b ON b.disciplina_id = tmp.course AND b.aluno_id = tmp.userid 
GROUP BY b.disciplina_id, b.aluno_id) AS var34
ON var34.disciplina_id = base.disciplina_id AND var34.aluno_id = base.aluno_id
where base.semestre = '2021.2'
ORDER BY
base.curso, base.semestre, base.periodo, base.disciplina_nome, base.aluno_nome
--	)
