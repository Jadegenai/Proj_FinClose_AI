class few_shot_settings:

    @staticmethod
    def get_prefix():
        return f"""
        You are an agent designed to interact with a Snowflake with schema detail in Snowflake querying about Company Invoice Data. You have to write syntactically correct Snowflake sql query based on a users question.
        No matter what the user asks remember your job is to produce relevant SQL and only include the SQL, not the through process. So if a user asks to display something, you still should just produce SQL.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        If you don't know the answer, provide what you think the sql should be but do not make up code if a column isn't available. Use snowflake aggregate functions like SUM, MIN, MAX, etc. if user ask to find total, minimum or maximum.
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
                            FROM DB_DEV.SC_COINBASE.INVOICE
                            GROUP BY ALL 
                            ORDER BY 2 DESC
                            LIMIT 10;''',
            },
            {
                "input": "Which are the top 10 high volume suppliers ?",
                "sql_cmd": '''SELECT VENDOR_NAME, COUNT(INVOICE_ID) AS NUMBER_OF_INVOICE
                            FROM DB_DEV.SC_COINBASE.INVOICE
                            GROUP BY ALL 
                            ORDER BY 2 DESC
                            LIMIT 10;''',
            },
            {
                "input": "How much outstanding balance is left to be paid ?",
                "sql_cmd": '''SELECT (SUM(INVOICE_AMOUNT) - SUM(AMOUNT_PAID)) AS OUTSTANDING_BALANCE
                            FROM DB_DEV.SC_COINBASE.INVOICE;''',
            },
            {
                "input": "Show the list of Vendors has balance left to be paid",
                "sql_cmd": '''SELECT VENDOR_NAME, (SUM(INVOICE_AMOUNT) - SUM(AMOUNT_PAID)) AS OUTSTANDING_BALANCE
                            FROM DB_DEV.SC_COINBASE.INVOICE
                            GROUP BY ALL 
                            HAVING OUTSTANDING_BALANCE > 0
                            ORDER BY 2 DESC;''',
            },
            {
                "input": "How many invoices were created manually ?",
                "sql_cmd": '''SELECT SOURCE AS INVOICE_TYPE, COUNT(INVOICE_ID) AS NUMBER_OF_INVOICE
                            FROM DB_DEV.SC_COINBASE.INVOICE
                            WHERE SOURCE = ''Manual Invoice Entry''
                            GROUP BY ALL;''',
            },
            {
                "input": "How many invoices were created by BOT ?",
                "sql_cmd": '''SELECT SOURCE AS INVOICE_TYPE, COUNT(INVOICE_ID) AS NUMBER_OF_INVOICE
                            FROM DB_DEV.SC_COINBASE.INVOICE
                            WHERE SOURCE = ''AUTOBOT PAYABLES''
                            GROUP BY ALL;''',
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
