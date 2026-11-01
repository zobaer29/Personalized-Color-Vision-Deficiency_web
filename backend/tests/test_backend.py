import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app
from app.api import routes
from app.core.predictor import CVDRecolorPredictor
from app.core.page_filter import page_filter
from app.core.screening import score
from app.core.model import create_model

def image_data(size=(3,1)):
    buffer=io.BytesIO();Image.new('RGB',size,(140,90,20)).save(buffer,format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

class BackendTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app)
    def test_untrained_is_not_served(self):
        with patch.object(routes.predictor,'weights_loaded',False):
            self.assertEqual(self.client.post('/api/v1/filter',json={'condition_id':1}).status_code,503)
            self.assertEqual(self.client.get('/health').json()['status'],'checkpoint_required')
    def test_request_validation(self):
        for payload in [{'condition_id':-1},{'condition_id':3},{'condition_id':0,'severity':1.1}]:
            self.assertEqual(self.client.post('/api/v1/filter',json=payload).status_code,422)
        for extra in [{'return_format':'gif'},{'condition':'invalid'},{'condition_id':0,'condition':'tritan'}]:
            self.assertEqual(self.client.post('/api/v1/recolor',json={'image_base64':'x',**extra}).status_code,422)
    def test_missing_plates_do_not_diagnose(self):
        with patch('app.core.screening.catalog',return_value=[]):
            self.assertFalse(self.client.get('/api/v1/plates').json()['available'])
            self.assertEqual(self.client.post('/api/v1/diagnose',json={'answers':[{'plate_id':'fake','user_answer':'12'}]}).status_code,503)
    def test_answer_integrity(self):
        fixture=[dict(id='p1',options=['12','Nothing'],normal_answer='12')]
        with patch('app.core.screening.catalog',return_value=fixture):
            good=[dict(plate_id='p1',user_answer='12')]
            self.assertEqual(score(good)['errors'],0)
            self.assertFalse(score(good)['diagnosed'])
            for bad in [good+good,[dict(plate_id='fake',user_answer='12')],[dict(plate_id='p1',user_answer='99')]]:
                with self.assertRaises(ValueError):score(bad)
    def test_checkpoint_and_small_images(self):
        # Synthetic fixture only tests plumbing, never production weights.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.pt'
            model=create_model('Restormer_RDB')
            torch.save({'model_name':'Restormer_RDB','model_state_dict':model.state_dict()},path)
            predictor=CVDRecolorPredictor(str(path),device='cpu')
            self.assertTrue(predictor.weights_loaded)
            data=image_data()
            a=predictor.recolor(data,'protan',.5,'webp')
            b=predictor.recolor(data,'protan',.5,'jpeg')
            self.assertTrue(a['recolored_base64'].startswith('data:image/webp'))
            self.assertTrue(b['recolored_base64'].startswith('data:image/jpeg'))
            self.assertFalse(b['cached'])
            self.assertTrue(predictor.recolor(data,'protan',.5,'jpeg')['cached'])
            with self.assertRaises(Exception):predictor.recolor('not base64','protan')
            filt=page_filter(predictor,1,0)
            np.testing.assert_allclose(np.array(filt['matrix']).reshape(4,5),np.eye(4,5),atol=1e-7)
            torch.save({'model_name':'Modified_Conditional_Restormer','model_state_dict':model.state_dict()},path)
            self.assertFalse(predictor.load_weights(str(path)))
            with self.assertRaises(RuntimeError):predictor.recolor(data,'protan')

if __name__=='__main__':unittest.main()
