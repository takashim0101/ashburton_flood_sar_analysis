# Use Node 20
FROM node:20-bullseye

WORKDIR /app

# Copy package.json and lockfile
COPY package*.json ./

# Install all dependencies
RUN npm install

# Copy project files
COPY . .

# Expose Vite's default port
EXPOSE 3000

# Start dev server with hot reload
CMD ["npm", "run", "dev", "--", "--host"]

