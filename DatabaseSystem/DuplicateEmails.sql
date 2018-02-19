/*
Write a SQL query to find all duplicate emails in a table named Person.
*/
SELECT EMAIL
FROM (  SELECT Email,COUNT(Email) count_email
        FROM Person
        GROUP BY Email) a
WHERE count_email >= 2