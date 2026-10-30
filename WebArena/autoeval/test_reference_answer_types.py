import unittest
from autoeval.evaluator import Evaluator

class Client:
    def one_step_chat(self,prompt,system_msg):
        self.prompt=prompt;self.system=system_msg
        return 'Thoughts: checked\nStatus: "failure"',None

class ReferenceTypesTest(unittest.TestCase):
    def evaluate(self,reference):
        client=Client()
        result,_,_=Evaluator({'mock':client}).eval_text({'response':'response'},'mock',reference)
        self.assertEqual(result['status'],'failure')
        return client
    def test_list_keeps_all_references(self):
        client=self.evaluate(['first review','second review'])
        self.assertIn("['first review', 'second review']",client.prompt)
        self.assertIn('contains the reference answer',client.system)
    def test_na_string_uses_special_case(self):
        self.assertIn('no relevant information',self.evaluate(' N/A ').system)
    def test_normal_string_unchanged(self):
        self.assertIn('Reference Answer: expected answer',self.evaluate('expected answer').prompt)

if __name__=='__main__':unittest.main()
