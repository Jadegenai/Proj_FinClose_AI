class few_shot_settings:

    @staticmethod
    def get_prefix():
        return f"""
        You are an agent designed to interact with a Snowflake with schema detail in Snowflake querying about Company's employees' expenses, Customer Revenue and Account Balances. 
        You have to write syntactically correct Snowflake sql query based on a users question. 
        If a user asks to display something, you should produce SQL to display the result.
        Take 2024 as current year and 2023 as previous year. Dec, Jan, Feb as Winter and Jun, Jul, Aug as Summer. 
        If the user ask why expense is higher or lower in current year compared to previous year then 
        you need to generate a SQL query to extract the total expense amount, average expense amount, total number of transaction of both the years.
        If the user ask why revenue is higher in summer and as compared to winter then you need to extract the revenue data for that season only.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        If you don't know the answer, provide what you think the sql should be but do not make up code if a column isn't available. 
        Use snowflake aggregate functions like SUM, MIN, MAX, etc. if user ask to find total, minimum or maximum.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. 
        Few rules to follow are: 
        1. Always use column aliases as per example and metadata
        2. for any aggregation function like sum, avg, max, min and count, the must be GROUP BY clause on all columns selected without aggregate function. 
        3. preference is to use direct inner or left join. Avoid inner queries in WHERE clause.
        4. Strictly do not use inner query for such questions from user. Refer example for next quarter and previous quarter question.
        """

    @staticmethod
    def get_suffix():
        return """Question: {question}
        Context: {context}

        SQL_cmd: ```sql ``` \n

        """, ["question", "context"]

    @staticmethod
    def get_examples():
        examples = [
            {
                "input": "Which are the top 10 high value suppliers ?",
                "sql_cmd": '''SELECT VENDOR_NAME, SUM(INVOICE_AMOUNT) AS TOTAL_INVOICE_AMOUNT
                            FROM DEMO_DB.SC_FINCLOSE.INVOICE
                            GROUP BY ALL 
                            ORDER BY 2 DESC
                            LIMIT 10;''',
            },
            {
                "input": "Which are the top 10 high volume suppliers ?",
                "sql_cmd": '''SELECT VENDOR_NAME, COUNT(INVOICE_ID) AS NUMBER_OF_INVOICE
                            FROM DEMO_DB.SC_FINCLOSE.INVOICE
                            GROUP BY ALL 
                            ORDER BY 2 DESC
                            LIMIT 10;''',
            },
            {
                "input": "How much outstanding balance is left to be paid ?",
                "sql_cmd": '''SELECT (SUM(INVOICE_AMOUNT) - SUM(AMOUNT_PAID)) AS OUTSTANDING_BALANCE
                            FROM DEMO_DB.SC_FINCLOSE.INVOICE;''',
            },
            {
                "input": "Show the list of Vendors has balance left to be paid",
                "sql_cmd": '''SELECT VENDOR_NAME, (SUM(INVOICE_AMOUNT) - SUM(AMOUNT_PAID)) AS OUTSTANDING_BALANCE
                            FROM DEMO_DB.SC_FINCLOSE.INVOICE
                            GROUP BY ALL 
                            HAVING OUTSTANDING_BALANCE > 0
                            ORDER BY 2 DESC;''',
            },
            {
                "input": "How many invoices were created manually ?",
                "sql_cmd": '''SELECT SOURCE AS INVOICE_TYPE, COUNT(INVOICE_ID) AS NUMBER_OF_INVOICE
                            FROM DEMO_DB.SC_FINCLOSE.INVOICE
                            WHERE SOURCE = ''Manual Invoice Entry''
                            GROUP BY ALL;''',
            },
            {
                "input": "How many invoices were created by BOT ?",
                "sql_cmd": '''SELECT SOURCE AS INVOICE_TYPE, COUNT(INVOICE_ID) AS NUMBER_OF_INVOICE
                            FROM DEMO_DB.SC_FINCLOSE.INVOICE
                            WHERE SOURCE = ''AUTOBOT PAYABLES''
                            GROUP BY ALL;''',
            },
            {
                "input": "Show the total employee expense amount per year for the last two years",
                "sql_cmd": '''SELECT YEAR(INVOICE_DATE) AS INVOICE_YEAR, SUM(PAYMENT_AMOUNT) AS TOTAL_EXPENSES_AMOUNT 
                            FROM DEMO_DB.SC_FINCLOSE.EMPLOYEE_EXPENSES 
                            GROUP BY ALL
                            ORDER BY 1 DESC
                            LIMIT 2;''',
            },
            {
                "input": "Show the average employee expense amount per year for the last two years",
                "sql_cmd": '''SELECT YEAR(INVOICE_DATE) AS INVOICE_YEAR, AVG(PAYMENT_AMOUNT) AS AVERAGE_EXPENSES_AMOUNT 
                            FROM DEMO_DB.SC_FINCLOSE.EMPLOYEE_EXPENSES 
                            GROUP BY ALL
                            ORDER BY 1 DESC
                            LIMIT 2;''',
            },
            {
                "input": "Show the number of employee expense invoices per year for the last two years",
                "sql_cmd": '''SELECT YEAR(INVOICE_DATE) AS INVOICE_YEAR, COUNT(INVOICE_NUMBER) AS NUMBER_OF_INVOICES
                            FROM DEMO_DB.SC_FINCLOSE.EMPLOYEE_EXPENSES 
                            GROUP BY ALL
                            ORDER BY 1 DESC
                            LIMIT 2;''',
            },
            {
                "input": "Why employee expense amount is higher in current year as compared to previous year ?",
                "sql_cmd": '''SELECT 
                                    YEAR(INVOICE_DATE) AS INVOICE_YEAR, 
                                    SUM(PAYMENT_AMOUNT) AS TOTAL_EXPENSES_AMOUNT, 
                                    AVG(PAYMENT_AMOUNT) AS AVERAGE_EXPENSES_AMOUNT, 
                                    COUNT(INVOICE_NUMBER) AS NUMBER_OF_INVOICES
                                FROM DEMO_DB.SC_FINCLOSE.EMPLOYEE_EXPENSES 
                                WHERE YEAR(INVOICE_DATE) IN (YEAR(CURRENT_DATE), YEAR(CURRENT_DATE)-1)
                                GROUP BY YEAR(INVOICE_DATE)
                                ORDER BY 1 DESC;''',
            },
            {
                "input": "Why customer revenue is higher in summer season as compared to winter season ?",
                "sql_cmd": '''SELECT 
                                CASE 
                                    WHEN MONTHNAME(YEAR_MONTH) IN ('Jun', 'Jul', 'Aug') THEN 'SUMMER'
                                    WHEN MONTHNAME(YEAR_MONTH) IN ('Dec', 'Jan', 'Feb') THEN 'WINTER'
                                END AS SEASON_NAME, 
                                CUSTOMER_NAME,
                                SUM(REVENUE_AMOUNT) AS TOTAL_REVENUE_AMOUNT
                                FROM DEMO_DB.SC_FINCLOSE.CUSTOMER_REVENUE
                                WHERE SEASON_NAME IS NOT NULL
                                GROUP BY SEASON_NAME, CUSTOMER_NAME
                                ORDER BY 3 DESC,1,2;''',
            },
            {
                "input": "Show year wise total balance for each of the category of GL Account",
                "sql_cmd": '''SELECT PERIOD_YEAR, CATEGORY, SUM(BALANCE) AS TOTAL_BALANCE 
                                FROM DEMO_DB.SC_FINCLOSE.GL_ACCOUNT_BALANCES_YEAR
                                GROUP BY 1, 2
                            ORDER BY 1 DESC, 2''',
            }
        ]
        return examples

    @staticmethod
    def get_example_template():
        template = """
        Input: {input}
        SQL_cmd: {sql_cmd}\n
        """
        example_variables = ["input", "sql_cmd"]
        return template, example_variables
