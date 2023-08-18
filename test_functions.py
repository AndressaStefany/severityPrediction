import unittest
import pandas as pd
from baseline_functions import filter_bug_severity, create_binary_feature, remove_urls_and_codes, remove_code_snippets

class TestURLRemoval(unittest.TestCase):
    def setUp(self):
        # Sample data for URL removal
        self.url_data = {'col_name': [
            'This is a text with a URL: http://example.com',
            'Another text with a URL: https://www.example.org',
            'No URL in this text',
            'URL again: http://example.org'
        ]}

        # Sample data for bug report functions
        self.bug_data = {
            '_id': [1, 2, 3, 4, 5],
            'bug_id': [101, 102, 103, 104, 105],
            'description': [
                'This is a critical bug',
                'Normal issue',
                'Another critical problem',
                'Enhancement request',
                'Major bug found'
            ],
            'bug_severity': ['critical', 'normal', 'critical', 'enhancement', 'major']
        }
        
        self.code_text = """
        This is some text with a code snippet:

        ```python
        def hello():
            print("Hello, world!")
        ```
        
        And here's another code snippet: `x = 42`.

        ```javascript
        console.log('Hello from JavaScript');
        ```
        """
        
    # def test_filter_bug_severity(self):
    #     bug_reports = pd.DataFrame(self.bug_data)
    #     filtered_reports = filter_bug_severity(bug_reports)

    #     expected_data = {
    #         '_id': [1, 3, 5],
    #         'bug_id': [101, 103, 105],
    #         'description': [
    #             'This is a critical bug',
    #             'Another critical problem',
    #             'Major bug found'
    #         ],
    #         'bug_severity': ['critical', 'critical', 'major']
    #     }
    #     expected_result = pd.DataFrame(expected_data)
        
    #     pd.testing.assert_frame_equal(filtered_reports['_id'], expected_result['_id'])
    #     pd.testing.assert_frame_equal(filtered_reports['bug_id'], expected_result['bug_id'])
    #     pd.testing.assert_frame_equal(filtered_reports['description'], expected_result['description'])

    def test_create_binary_feature(self):
        bug_reports = pd.DataFrame(self.bug_data)
        bug_reports_with_binary = create_binary_feature(bug_reports)

        expected_data = {
            '_id': [1, 2, 3, 4, 5],
            'bug_id': [101, 102, 103, 104, 105],
            'description': [
                'This is a critical bug',
                'Normal issue',
                'Another critical problem',
                'Enhancement request',
                'Major bug found'
            ],
            'bug_severity': ['critical', 'normal', 'critical', 'enhancement', 'major'],
            'binary_severity': [1, 0, 1, 0, 1]
        }
        expected_result = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(bug_reports_with_binary, expected_result)
    
    def test_url_removal(self):
        data = {'col_name': [
            'This is a text with a URL: http://example.com',
            'Another text with a URL: https://www.example.org',
            'No URL in this text',
            'URL again: http://example.org'
        ]}
        expected_result = pd.DataFrame({
            'col_name': [
                'This is a text with a URL: ',
                'Another text with a URL: ',
                'No URL in this text',
                'URL again: '
            ]
        })

        input_df = pd.DataFrame(data)
        output_df = remove_urls_and_codes(input_df.copy(), 'col_name')

        pd.testing.assert_frame_equal(output_df, expected_result)
        
    def test_remove_code_snippets(self):        
        expected_output = """
        This is some text with a code snippet:

        CODE
        
        And here's another code snippet: .

        CODE
        """
        
        cleaned_text = remove_code_snippets(self.code_text)
        
        self.assertEqual(cleaned_text, expected_output)
        
if __name__ == '__main__':
    unittest.main()
