class CustomStatuspalWidget {
    constructor(
      widgetBtnId,
      widgetDataContainerId,
      statuspalUrl,
      containerPosition,
      devMode,
      sessionStorageKey
    ) {
      // button settings
      this.widgetBtn = document.getElementById(widgetBtnId);
      this.widgetBtn.classList.add('sp-custom-widget');
      this.widgetBtn.setAttribute('aria-controls', widgetDataContainerId);
      this.widgetBtn.setAttribute('aria-expanded', 'false');
      this.widgetBtn.addEventListener('click', this.toggleDisplay.bind(this));
  
      // Create a span element for the counter
      const counterSpan = document.createElement('span');
      counterSpan.classList.add('counter');
      this.widgetBtn.appendChild(counterSpan);
  
      // add counter element and unread class if new incident happen
      document.addEventListener('statuspalNotification', () => {
        this.toggleUnreadStatus();
      });
  
      // container settings
      this.widgetDataContainer = document.getElementById(widgetDataContainerId);
      this.widgetDataContainer.classList.add('sp-custom-data-container');
      this.widgetDataContainer.style.display = 'none';
      this.widgetDataContainer.classList.add(containerPosition);
      this.widgetDataContainer.setAttribute('aria-hidden', 'true');
  
      // close container if clicking outside
      document.addEventListener('click', this.blurwidgetDataContainer.bind(this));
  
      // other settings
      this.sessionStorageKey = sessionStorageKey;
      this.devMode = devMode;
      this.statuspalUrl = statuspalUrl;
  
      this.getStatuspalData();
      this.setCountInButton();
    }
  
    isDevMode() {
      return this.devMode;
    }
  
    toggleUnreadStatus() {
      if (this.widgetBtn.classList.contains('sp-has-unread')) {
        this.setReadStatus();
      } else {
        this.setUnreadStatus();
      }
    }
  
    setReadStatus() {
      this.widgetBtn.classList.remove('sp-has-unread');
    }
  
    setUnreadStatus() {
      this.widgetBtn.classList.add('sp-has-unread');
    }
  
    isWidgetDataContainerOpen() {
      const isOpen =
        this.widgetDataContainer.style.display === 'block' ||
        !this.widgetDataContainer.style.display.length;
      return isOpen;
    }
  
    toggleDisplay() {
      if (this.isWidgetDataContainerOpen()) {
        this.closeWidgetDataContainer();
      } else {
        this.openWidgetDataContainer();
      }
      // Update aria-expanded for the button
      const isExpanded = this.isWidgetDataContainerOpen();
      this.widgetBtn.setAttribute('aria-expanded', isExpanded.toString());
    }
  
    closeWidgetDataContainer() {
      this.widgetDataContainer.style.display = 'none';
      this.widgetDataContainer.setAttribute('aria-hidden', 'true');
    }
  
    openWidgetDataContainer() {
      this.widgetDataContainer.style.display = 'block';
      this.widgetDataContainer.setAttribute('aria-hidden', 'false');
      this.statuspalData.viewed = true;
      this.setReadStatus();
      this.saveJsonToSessionStorage(this.statuspalData);
    }
  
    createEventDivs() {
      this.statuspalData.incidents.forEach(incident => {
        this.createEventDiv(incident, 'incident');
      });
  
      this.statuspalData.maintenances.forEach(maintenance => {
        this.createEventDiv(maintenance, 'maintenance');
      });
  
      this.statuspalData.upcoming_maintenances.forEach(upcomingMaintenance => {
        this.createEventDiv(upcomingMaintenance, 'upcoming maintenance');
      });
    }
  
    createEventDiv(event, eventName) {
      const eventDiv = document.createElement('div');
      this.widgetDataContainer.appendChild(eventDiv);
  
      const spEvent = 'sp-event';
      const spEventType = `sp-event-type-${event.type}`;
      eventDiv.classList.add(spEvent, spEventType);
  
      const eventType = document.createElement('p');
      eventType.classList.add('event-type');
      eventDiv.appendChild(eventType);
      eventType.innerText = `${event.type} ${eventName}`;
  
      const eventTitle = document.createElement('strong');
      eventDiv.appendChild(eventTitle);
      eventTitle.innerText = event.title;
  
      const eventUpdated = document.createElement('p');
      eventUpdated.classList.add('.event-updated-time');
      eventDiv.appendChild(eventUpdated);
      eventUpdated.innerText = `Last updated: ${this.formatDate(
        event.updated_at
      )}`;
  
      const readMore = document.createElement('a');
      eventDiv.appendChild(readMore);
      readMore.href = event.url;
      readMore.innerText = 'Read more';
    }
  
    createfallbackEventDiv () {
      const eventDiv = document.createElement('div');
  
      const spEvent = 'sp-event';
      const spEventType = `sp-event-type-fallback`;
      eventDiv.classList.add(spEvent, spEventType);
  
      const eventTitle = document.createElement('strong');
      eventDiv.appendChild(eventTitle);
      eventTitle.innerText = 'There are no ongoing events';
  
      this.widgetDataContainer.appendChild(eventDiv);
    }
  
    getStatuspalData() {
      // If we're in dev mode we use test data directly
      if (this.isDevMode()) {
        this.statuspalData = this.testData;
        this.statuspalData.viewed = false; // set data as unread
        console.log(this.statuspalData, 'Test data is loaded');
        this.getOpenIncidentCount();
        if (this.hasEvents()) {
          this.createEventDivs();
        } else {
          this.createfallbackEventDiv();
        }
      } else {
        // Normal behavior: check session storage or fetch new data
        const statuspalData = this.getJsonFromSessionStorage();
        if (Object.keys(statuspalData).length === 0) {
          this.fetchStatuspalData();
        } else {
          this.statuspalData = statuspalData;
          if (this.hasEvents()) {
            this.createEventDivs();
          } else {
            this.createfallbackEventDiv();
          }
          if (!this.isStatuspalDataViewed()) {
            this.dispatchStatuspalNotification();
          }
        }
      }
    }
    getOpenIncidentCount() {
      const incidents = Array.isArray(this.statuspalData.incidents)
        ? this.statuspalData.incidents.length
        : 0;
      const maintenances = Array.isArray(this.statuspalData.maintenances)
        ? this.statuspalData.maintenances.length
        : 0;
      const upcomingMaintenances = Array.isArray(
        this.statuspalData.upcoming_maintenances
      )
        ? this.statuspalData.upcoming_maintenances.length
        : 0;
  
      return incidents + maintenances + upcomingMaintenances;
    }
  
    setCountInButton() {
      const count = this.getOpenIncidentCount();
      const countContainer = this.widgetBtn.querySelector('.counter');
  
      if (count > 0) {
        countContainer.textContent = count;
        countContainer.style.display = ''; // Show the counter
      } else {
        countContainer.style.display = 'none'; // Hide the counter
        this.setReadStatus();
      }
    }
  
    fetchStatuspalData() {
      const statuspalUrl = this.statuspalUrl;
      fetch(statuspalUrl)
        .then(response => response.json())
        .then(data => {
          this.statuspalData = data;
          this.saveJsonToSessionStorage(data);
          if (this.hasEvents()) {
            this.createEventDivs();
            this.setCountInButton();
            this.dispatchStatuspalNotification();
          }
        })
        .catch(error => {
          console.error('Error fetching statuspal data:', error);
        });
    }
  
    hasEvents() {
      return (
        this.statuspalData.incidents.length > 0 ||
        this.statuspalData.maintenances.length > 0 ||
        this.statuspalData.upcoming_maintenances.length > 0
      );
    }
  
    isStatuspalDataViewed() {
      return this.statuspalData && this.statuspalData.viewed === true;
    }
  
    saveJsonToSessionStorage(value) {
      sessionStorage.setItem(this.sessionStorageKey, JSON.stringify(value));
    }
  
    getJsonFromSessionStorage() {
      const storedValue = sessionStorage.getItem(this.sessionStorageKey);
      return storedValue ? JSON.parse(storedValue) : {};
    }
  
    formatDate(dateString) {
      const date = new Date(dateString);
      const day = String(date.getDate()).padStart(2, '0');
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const year = String(date.getFullYear());
      const formattedDate = `${day}-${month}-${year}`;
      return formattedDate;
    }
  
    dispatchStatuspalNotification() {
      document.dispatchEvent(new CustomEvent('statuspalNotification'));
    }
  
    blurwidgetDataContainer(event) {
      const isClickInside = this.widgetDataContainer.contains(event.target);
      const isInsideBtn = this.widgetBtn.contains(event.target);
      if (this.isWidgetDataContainerOpen() && !isClickInside && !isInsideBtn) {
        this.setReadStatus();
        this.widgetDataContainer.style.display = 'none';
        this.widgetDataContainer.setAttribute('aria-hidden', 'true');
      }
    }
  
    testData = {
      status_page: {
        url: 'example.com',
        time_zone: 'CET',
        subdomain: 'example-data',
        name: 'Example',
        current_incident_type: 'major'
      },
      services: [
        {
          name: 'Monitoring service',
          id: 0,
          current_incident_type: 'major',
          children: [{}]
        }
      ],
      incidents: [
        {
          id: 89,
          inserted_at: '2022-01-01T00:00:00',
          updated_at: '2022-01-01T00:00:00',
          title: 'We are having an incident with the DB connection',
          starts_at: '2022-01-01T00:00:00',
          ends_at: '2022-01-01T00:00:00',
          type: 'major',
          service_ids: [1],
          updates: [
            {
              id: 89,
              inserted_at: '2022-01-01T00:00:00',
              updated_at: '2022-01-01T00:00:00',
              posted_at: '2022-01-01T00:00:00',
              type: 'issue',
              description: 'We have detected an issue with our CDN',
              description_html: '<p>We have detected an issue with our CDN</p>',
              translations: {
                en: {
                  description: 'English description'
                },
                es: {
                  description: 'Spanish description'
                }
              },
              subscribers_notified_at: '2022-01-01T00:00:00',
              tweet: true
            }
          ],
          url: 'https://status.example.com/incidents/123',
          translations: {
            en: {
              title: 'English Title'
            },
            es: {
              title: 'Spanish Title'
            }
          }
        }
      ],
      maintenances: [
        {
          id: 89,
          inserted_at: '2022-01-01T00:00:00',
          updated_at: '2022-01-01T00:00:00',
          title: 'We are doing maintenance with the DB connection',
          starts_at: '2022-01-01T00:00:00',
          ends_at: '2022-01-01T00:00:00',
          type: 'minor',
          service_ids: [1],
          updates: [
            {
              id: 89,
              inserted_at: '2022-01-01T00:00:00',
              updated_at: '2022-01-01T00:00:00',
              posted_at: '2022-01-01T00:00:00',
              type: 'issue',
              description: 'We have detected an issue with our CDN',
              description_html: '<p>We have detected an issue with our CDN</p>',
              translations: {
                en: {
                  description: 'English description'
                },
                es: {
                  description: 'Spanish description'
                }
              },
              subscribers_notified_at: '2022-01-01T00:00:00',
              tweet: true
            }
          ],
          url: 'https://status.example.com/incidents/123',
          translations: {
            en: {
              title: 'English Title'
            },
            es: {
              title: 'Spanish Title'
            }
          }
        }
      ],
      upcoming_maintenances: [
        {
          id: 89,
          inserted_at: '2022-01-01T00:00:00',
          updated_at: '2022-01-01T00:00:00',
          title: 'We have a upcoming maintenanced planned',
          starts_at: '2022-01-01T00:00:00',
          ends_at: '2022-01-01T00:00:00',
          type: 'scheduled',
          service_ids: [1],
          updates: [
            {
              id: 89,
              inserted_at: '2022-01-01T00:00:00',
              updated_at: '2022-01-01T00:00:00',
              posted_at: '2022-01-01T00:00:00',
              type: 'issue',
              description: 'We have detected an issue with our CDN',
              description_html: '<p>We have detected an issue with our CDN</p>',
              translations: {
                en: {
                  description: 'English description'
                },
                es: {
                  description: 'Spanish description'
                }
              },
              subscribers_notified_at: '2022-01-01T00:00:00',
              tweet: true
            }
          ],
          url: 'https://status.example.com/incidents/123',
          translations: {
            en: {
              title: 'English Title'
            },
            es: {
              title: 'Spanish Title'
            }
          }
        }
      ],
      viewed: false
    };
  }
  
  // eslint-disable-next-line no-unused-vars
  function initializeCustomStatuspalWidget(config) {
    window.customStatuspalWidgetInstance = new CustomStatuspalWidget(
      config.widgetBtnId,
      config.widgetDataContainerId,
      config.statuspalUrl,
      config.containerPosition,
      config.devMode,
      config.sessionStorageKey || 'statuspalData',
    );
  
    window.dispatchEvent(new CustomEvent('customStatuspalWidgetInitialized'));
  }  