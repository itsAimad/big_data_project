FROM node:alpine

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm install

COPY . .

CMD ["node","index.js"]