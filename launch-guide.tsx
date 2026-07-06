import React from 'react';
import { Alert, AlertTitle } from '@/components/ui/alert';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const LaunchGuide = () => {
  return (
    <div className="space-y-6 w-full max-w-4xl">
      <Alert>
        <AlertTitle>Prerequisites</AlertTitle>
        <p>Ensure you have Python 3.11+ and Homebrew installed on your MacBook M1</p>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle>1. Initial Setup</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-gray-100 p-4 rounded">
            <pre className="whitespace-pre-wrap">
              # Clone repository and enter directory
              git clone [your-repo-url]
              cd jarvis

              # Make install script executable
              chmod +x scripts/install.sh

              # Run installation script
              ./scripts/install.sh
            </pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>2. Configure Jarvis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p>Edit config/jarvis_config.json to match your setup:</p>
          <div className="bg-gray-100 p-4 rounded">
            <ul className="list-disc list-inside space-y-2">
              <li>Set correct camera_index</li>
              <li>Configure automation devices</li>
              <li>Adjust audio settings</li>
              <li>Set memory retention period</li>
            </ul>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>3. Launch Jarvis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-gray-100 p-4 rounded">
            <pre className="whitespace-pre-wrap">
              # Activate virtual environment
              source venv/bin/activate

              # Start Jarvis web interface
              python -m jarvis.web.server
            </pre>
          </div>
          <p>Access the dashboard at: http://localhost:8000</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>4. Verify Components</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-green-50 rounded">
              <h4 className="font-semibold">Vision</h4>
              <p>Camera preview should appear</p>
            </div>
            <div className="p-3 bg-blue-50 rounded">
              <h4 className="font-semibold">Audio</h4>
              <p>Microphone indicator active</p>
            </div>
            <div className="p-3 bg-purple-50 rounded">
              <h4 className="font-semibold">Memory</h4>
              <p>Database initialized</p>
            </div>
            <div className="p-3 bg-yellow-50 rounded">
              <h4 className="font-semibold">Automation</h4>
              <p>Devices connected</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LaunchGuide;
