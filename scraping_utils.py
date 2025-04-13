import requests
from bs4 import BeautifulSoup
import json


def parse_page(url: str) -> str:
    """
    Parse the content of a web page and extract its text.

    Args:
        url (str): The URL of the web page to parse.

    Returns:
        str: The extracted text content of the web page.

    Raises:
        requests.RequestException: If there's an error fetching the page.
    """
    
    try:
        response = requests.get(url.strip())
        response.raise_for_status()
        # Check if the response code is 403 (Forbidden), if so, return None
        if response.status_code == 403:
            print(f"Skipping {url}: Forbidden (403)")
            return None
        
        soup = BeautifulSoup(response.content, 'lxml')
        return soup.body.get_text()
    # except requests.RequestException as e:
    #     raise requests.RequestException(f"Error fetching the page: {e}")
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (like 403, 404, etc.)
        print(f"HTTP Error for {url}: {e}")
        return None  # Return None to indicate failure
    
    except requests.exceptions.RequestException as e:
        # Handle other errors like connection issues
        print(f"Error fetching {url}: {e}")
        return None



def scrape_lawyer_profile(url):
    # Fetch the webpage
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None
    
    # Parse the webpage content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Initialize fixed structure with placeholders
    data = {
        "Name": "",
        "Job Title": "",
        "Email": "",
        "Phone": "",
        "Firm Name": "Davis Polk",
        "Description": [],
        "Capabilities": [],
        "Experience": [],
        "Education": [],
        "Languages": [],
        "Clerkships": [],
        "Qualifications": [],
        "News": [],
        "Insights": [],
        "Recognition": []

    }

    # Extract Name and Description from meta tags
    data["Name"] = soup.find('meta', {'property': 'og:title'})['content']
  
    # Extract Description
    job_description = soup.find('div', class_ = 'field-name--field_snippet field-item')
    if job_description:
      data["Description"].append(job_description.get_text(strip=True))

    # Extract Descripption
    job_description = soup.find('div', class_ = 'field-name--field_biography text-formatted field-item')
    if job_description:
      data["Description"].append(job_description.get_text(strip=True))
    
    # Extract Job Title
    job_title_tag = soup.find('div', class_='field-name--field_job_title')
    if job_title_tag:
        data["Job Title"] = job_title_tag.get_text(strip=True)
    
    # Extract JSON-LD data for Email, Phone, Languages
    script_tag = soup.find('script', type='application/ld+json')
    if script_tag:
        try:
            json_data = json.loads(script_tag.string)
            person_data = json_data["@graph"][0]
            data["Email"] = person_data.get("email", "")
            data["Phone"] = person_data.get("telephone", "")
            data["Languages"] = person_data.get("knowsLanguage", [])
            data["Firm Name"] = person_data.get("worksFor", {}).get("name", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Extract Experience Highlights
    experience_section = soup.find('div', class_='lawyer--experience-highlights')
    if experience_section:
        items = experience_section.find_all('li')
        data["Experience"] = [item.get_text(strip=True) for item in items]

    # Add Insights under Experience if present
    insights_section = soup.find('div', class_='lawyer--insights')
    if insights_section:
        insights_items = insights_section.find_all('article', class_='node-type-insights_article')
        for item in insights_items:
            title = item.find('h3', class_='node-title')
            teaser = item.find('div', class_='article--teaser')
            if title and teaser:
                data["Insights"].append(f"{title.get_text(strip=True)} - {teaser.get_text(strip=True)}")

    # Add News under Experience if present
    news_section = soup.find('div', class_='lawyer--news')
    if news_section:
        news_items = news_section.find_all('article', class_='node-type-news')
        for item in news_items:
            title = item.find('h3', class_='node-title')
            if title:
                data["News"].append(f"{title.get_text(strip=True)}")

    # Add Recognition under Experience if present
    recognition_section = soup.find('div', class_='lawyer--recognition')
    if recognition_section:
        recognition_items = recognition_section.find_all('p')
        for item in recognition_items:
            data["Recognition"].append(f"{item.get_text(strip=True)}")

    # Extract Capabilities (Two sections: desktop and accordion)
    capabilities_section = soup.find('div', class_='lawyer--sidebar-capabilities')
    if capabilities_section:
        desktop_items = capabilities_section.find_all('div', class_='field-item')
        for item in desktop_items:
            link = item.find('a')
            if link:
                data["Capabilities"].append(link.get_text(strip=True))

      # Extract Education
    education_section = soup.find('div', class_='lawyer--education')
    if education_section:
        # Find all degree elements and append them as single elements to the list
        degree_items = education_section.find_all('div', class_='degree')
        for degree in degree_items:
            data["Education"].append(degree.get_text(strip=True))

    # Extract Clerkships
    clerkships_section = soup.find('div', class_='lawyer--clerkships')
    if clerkships_section:
        # Find all clerkship items and append them as single elements to the list
        clerkship_items = clerkships_section.find_all('div', class_='field-item')
        for clerkship in clerkship_items:
            data["Clerkships"].append(clerkship.get_text(strip=True))



    # Lawyer Qualifications
    qualifications_section = soup.find('div', class_='lawyer--licenses')
    if qualifications_section:
        items = qualifications_section.find_all('li')
        data["Qualifications"] = [item.get_text(strip=True) for item in items]
        

    # Prior Experience
    p_exp = soup.find('div', class_='lawyer--work-experience')
    if p_exp:
        items = p_exp.find_all('li')
        for item in items:
            data["Experience"].append(f"Prior Experience: {item.get_text(strip=True)}")

    return data
