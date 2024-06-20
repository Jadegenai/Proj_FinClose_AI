class zero_shot_analyze_settings:
    
    @staticmethod
    def get_prompt_template(ask, metadata):
        return ("Based on ask wihtin a context, generated output from snowflake database base based on metadata are as follow. \n" +
        ",Ask: " + ask + "\n" +
        ",Metadata: " + metadata + "\n" + 
        "Question: {question} \n" +
        "Context: {context} \n\n" +
        "Result:"), ["question", "context"]