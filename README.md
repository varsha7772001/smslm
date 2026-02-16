**Android LLM Module (llama.cpp Integration)**

This Android library integrates llama.cpp for on-device LLM inference.

1️⃣ **Add JitPack Repository**

In your settings.gradle:

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}

2️⃣ **Add Dependency**

In your app/build.gradle:

implementation 'com.github.varsha7772001:smslm:main'

with your actual repo details.

🧠 **Clone Latest llama.cpp**

This project uses the official llama.cpp repository:

ggerganov/llama.cpp

Official GitHub:
https://github.com/ggerganov/llama.cpp

To clone latest manually:

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git pull origin master

Using Git Submodule (Recommended)

To always track latest llama.cpp inside this project:

**From project root:**

git submodule add https://github.com/ggerganov/llama.cpp.git cpp/llama.cpp
git submodule update --init --recursive


This will:

Add llama.cpp inside cpp/llama.cpp

Keep it version controlled


