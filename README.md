# 3D-content-retrieval

Welcome to our web application designed to implement a robust Content-Based 3D Models Retrieval system that enables efficient image search and management through visual features. Users can upload, download, delete, and categorize models into predefined classes. The system computes and displays shape descriptors for models, including Zernike moments, and Fourier descriptors, in addition to viewing the model in 3D. It supports a simple search to retrieve visually similar models, providing an intuitive and dynamic way to explore the Pottery dataset.

## Installation

To start off, clone this branch of the repo into your local:

```shell
git clone https://github.com/Samashi47/3D-content-retrieval.git
```

```shell
cd 3D-content-retrieval
```

### Backend

After cloning the project, create a virtual environment:

```shell
cd apps/api
```

**Windows**

```shell
py -3 -m venv .venv
```

**MacOS/Linus**

```shell
python3 -m venv .venv
```

Then, activate the env:

**Windows**

```shell
.venv\Scripts\activate
```

**MacOS/Linus**

```shell
. .venv/bin/activate
```

You can run the following command to install the dependencies:

```shell
pip3 install -r requirements.txt
```

After installing the dependencies, you should specify the mongodb connection string in the `.env` file:

```shell
touch .env
```

or:

```shell
cp .env.example .env
```

Then, open the `.env` file and add the following line:

```env
MONGO_URL=<url>
```

To be able to use the register and login functionalities, you need to generate jwt key and place it in a .ssh folder in the api directory:

```shell
openssl genrsa -out jwt-key 4096
```

to extract the corresponding public key:

```shell
openssl rsa -in jwt-key -pubout -out jwt-key.pub
```

then, move the keys to the .ssh folder:

```shell
mkdir .ssh
mv jwt-key .ssh
mv jwt-key.pub .ssh
```

also, you need to download the 3DPottery dataset from the following link:

```shell
http://www.ipet.gr/~akoutsou/benchmark/
```
And place the 3D Models and Thmbnail folders in the api\assets directory.
The assets directory should look like this:

```shell
├───assets
│   ├───3D Models
│   │   ├───Abstract
│   │   ├───Alabastron
│   │   ├───Amphora
│   │   ├───...
│   └───Thumbnails
│       ├───Abstract
│       ├───Alabastron
│       ├───Amphora
│       ├───...
```
Then, you can run the following command to start the backend:

```shell
python server.py
```

### Frontend

Open another terminal:

```shell
cd 3D-content-retrieval
```

```shell
cd apps/app
```

Then, run the following command to install the dependencies:

```shell
pnpm install
```

then, run the following command to start the frontend, if you have angular cli installed globally:

```shell
ng serve
```

if not, you can run the following command:

```shell
pnpm run ng serve
```

Then, open your browser and navigate to `http://localhost:4200/` to see the app running.
