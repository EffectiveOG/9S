// Enhanced dashboard JavaScript with charts and real-time updates
const Dashboard = {
    // WebSocket connection
    ws: null,
    // Chart objects
    charts: {},
    // Metric history
    metricHistory: {
        cpu: [],
        memory: [],
        gpu: []
    },
    
    init() {
        this.initWebSocket();
        this.initCharts();
        this.initEventHandlers();
    },
    
    initWebSocket() {
        this.ws = new WebSocket(`ws://${window.location.host}/ws`);
        this.ws.onmessage = (event) => this.handleWebSocketMessage(JSON.parse(event.data));
        this.ws.onclose = () => setTimeout(() => this.initWebSocket(), 1000);
    },
    
    initCharts() {
        // CPU Usage Chart
        this.charts.cpu = new Chart(
            document.getElementById('cpuChart').getContext('2d'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage %',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            }
        );
        
        // Memory Usage Chart
        this.charts.memory = new Chart(
            document.getElementById('memoryChart').getContext('2d'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Memory Usage %',
                        data: [],
                        borderColor: 'rgb(153, 102, 255)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            }
        );
    },
    
    initEventHandlers() {
        // Command console
        document.getElementById('commandForm').addEventListener('submit', (e) => {
            e.preventDefault();
            const command = document.getElementById('commandInput').value;
            try {
                const parsedCommand = JSON.parse(command);
                this.sendCommand(parsedCommand);
                document.getElementById('commandInput').value = '';
            } catch (e) {
                alert('Invalid JSON command format');
            }
        });
        
        // Scene controls
        document.querySelectorAll('.scene-button').forEach(button => {
            button.addEventListener('click', () => {
                this.activateScene(button.dataset.scene);
            });
        });
    },
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'metrics':
                this.updateMetrics(message.data);
                break;
            case 'command_response':
                this.handleCommandResponse(message.data);
                break;
            case 'error':
                this.handleError(message.data);
                break;
        }
    },
    
    updateMetrics(metrics) {
        // Update charts
        const timestamp = new Date().toLocaleTimeString();
        
        // Update CPU chart
        this.updateChart(this.charts.cpu, timestamp, metrics.system.cpu.usage_percent);
        
        // Update Memory chart
        this.updateChart(this.charts.memory, timestamp, metrics.system.memory.percent);
        
        // Update component status
        this.updateComponentStatus(metrics.components);
        
        // Update device status
        this.updateDeviceStatus(metrics.jarvis.devices);
    },
    
    updateChart(chart, label, value) {
        chart.data.labels.push(label);
        chart.data.datasets[0].data.push(value);
        
        // Keep only last 60 points
        if (chart.data.labels.length > 60) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update();
    },
    
    updateComponentStatus(components) {
        const statusDiv = document.getElementById('componentStatus');
        statusDiv.innerHTML = '';
        
        Object.entries(components).forEach(([name, status]) => {
            const componentDiv = document.createElement('div');
            componentDiv.className = `component-status ${status.toLowerCase()}`;
            componentDiv.innerHTML = `
                <span class="component-name">${name}</span>
                <span class="component-indicator"></span>
            `;
            statusDiv.appendChild(componentDiv);
        });
    },
    
    updateDeviceStatus(devices) {
        const devicesDiv = document.getElementById('deviceStatus');
        devicesDiv.innerHTML = '';
        
        Object.entries(devices).forEach(([id, status]) => {
            const deviceDiv = document.createElement('div');
            deviceDiv.className = 'device-card';
            deviceDiv.innerHTML = `
                <h3>${id}</h3>
                <div class="device-status">
                    <span class="status-indicator ${status.power ? 'on' : 'off'}"></span>
                    ${this.formatDeviceStatus(status)}
                </div>
                <div class="device-controls">
                    ${this.getDeviceControls(id, status)}
                </div>
            `;
            devicesDiv.appendChild(deviceDiv);
        });
    },
    
    formatDeviceStatus(status) {
        let statusHtml = '';
        Object.entries(status).forEach(([key, value]) => {
            if (key !== 'type') {
                statusHtml += `<div>${key}: ${value}</div>`;
            }
        });
        return statusHtml;
    },
    
    getDeviceControls(id, status) {
        switch (status.type) {
            case 'light':
                return `
                    <button onclick="Dashboard.sendCommand({
                        type: 'device_control',
                        device: '${id}',
                        action: 'power_toggle'
                    })">Toggle Power</button>
                    <input type="range" min="0" max="100" value="${status.brightness || 0}"
                           onchange="Dashboard.sendCommand({
                               type: 'device_control',
                               device: '${id}',
                               action: 'set_brightness',
                               value: this.value
                           })">
                `;
            case 'tv':
                return `
                    <button onclick="Dashboard.sendCommand({
                        type: 'device_control',
                        device: '${id}',
                        action: 'power_toggle'
                    })">Toggle Power</button>
                    <select onchange="Dashboard.sendCommand({
                        type: 'device_control',
                        device: '${id}',
                        action: 'set_input',
                        value: this.value
                    })">
                        <option value="hdmi1">HDMI 1</option>
                        <option value="hdmi2">HDMI 2</option>
                        <option value="tv">TV</option>
                    </select>
                `;
            default:
                return `
                    <button onclick="Dashboard.sendCommand({
                        type: 'device_control',
                        device: '${id}',
                        action: 'power_toggle'
                    })">Toggle Power</button>
                `;
        }
    },
    
    sendCommand(command) {
        this.ws.send(JSON.stringify({
            type: 'command',
            data: command
        }));
    },
    
    activateScene(scene) {
        this.sendCommand({
            type: 'scene_control',
            action: 'activate',
            scene: scene
        });
    },
    
    handleCommandResponse(response) {
        if (response.status === 'accepted') {
            this.showNotification('Command accepted', 'success');
        } else {
            this.showNotification('Command failed', 'error');
        }
    },
    
    handleError(error) {
        this.showNotification(error.message, 'error');
    },
    
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        setTimeout(() => notification.remove(), 3000);
    }
};

// Initialize dashboard when document is ready
document.addEventListener('DOMContentLoaded', () => Dashboard.init());