import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_what_if_query_param_support(client):
    """Verify what-if page renders with query params."""
    response = client.get('/what-if?location=Delhi')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'URLSearchParams' in html
    assert 'sim-location' in html
    assert 'Delhi' in html

def test_cooling_stations_query_param_support(client):
    """Verify cooling stations page renders with city query param."""
    response = client.get('/cooling-stations?city=Hyderabad')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'URLSearchParams' in html
    assert 'city-filter-select' in html
    assert 'Hyderabad' in html

def test_alerts_query_param_support(client):
    """Verify alerts page renders with location, risk, role query params."""
    response = client.get('/alerts?location=Delhi&risk=High&role=worker')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'URLSearchParams' in html
    assert 'gen-city-select' in html
    assert 'gen-risk-select' in html
    assert 'gen-role-select' in html

def test_authority_query_param_support(client):
    """Verify authority map page renders with city query param."""
    response = client.get('/authority?city=Delhi')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'URLSearchParams' in html
    assert 'city-focus-select' in html

def test_institute_query_param_support(client):
    """Verify worksite page renders with institute_id query param."""
    response = client.get('/worksite?institute_id=inst_nhai_infra')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'URLSearchParams' in html
    assert 'institute-select' in html

def test_validation_query_param_support(client):
    """Verify validation case study explorer renders with case param."""
    response = client.get('/validation?case=case_khammam_2015')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'URLSearchParams' in html
    assert 'case-selector' in html
