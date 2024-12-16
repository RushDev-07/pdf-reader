from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import base64
import time

# Set up the Chrome driver
chrome_driver_path = 'C:\Webdrivers\chromedriver.exe'  # Replace with the actual path to chromedriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service)

# Open the main URL
main_url = 'https://maharera.maharashtra.gov.in/projects-search-result?page=0&project_division=6'  # Replace with the actual main URL
driver.get(main_url)
time.sleep(5)  # Wait for page to load

# Loop to find and download certificates
for i in range(1, 2):  # Adjust range as needed
    try:
        # Find the button with specific data-qstr and title
        button = driver.find_element(
            By.XPATH,
            f'//a[@data-hqstr="12--{i}"][contains(@title, "View Original Application")]'
        )
        
        # Scroll to the element and click
        driver.execute_script("arguments[0].scrollIntoView();", button)
        ActionChains(driver).move_to_element(button).click(button).perform()
        time.sleep(5)  # Wait for modal to open

        # Locate the base64 string from the embedded PDF
        pdf_data_element = driver.find_element(By.XPATH, '//object[@type="application/pdf"]')
        data_attribute = pdf_data_element.get_attribute('data')
        
        # Extract and decode the base64 content
        if data_attribute and 'base64,' in data_attribute:
            base64_string = data_attribute.split('base64,')[1]
            pdf_data = base64.b64decode(base64_string)
            
            # Save the PDF file
            with open(f"SOURCE_DOCUMENTS/applications_{i}.pdf", "wb") as pdf_file:
                pdf_file.write(pdf_data)
            print(f"Application {i} downloaded successfully.")
        
        # Close the modal (assuming there's a close button)
        close_button = driver.find_element(By.XPATH, '//button[@class="close-modal"]')
        close_button.click()
        time.sleep(1)  # Wait for modal to close

    except Exception as e:
        print(f"An error occurred with Applicaiton {i}: {e}")

# Close the driver
driver.quit()