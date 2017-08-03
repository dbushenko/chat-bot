# chatbot

A Clojure program which allows to talk to a virtual person. The training texts are located in folder 'training' and should be in the form

     question1?
	 answer1
     question2?
	 answer2
	 questionN?
	 answerN

All the sample (question or answer) should be on one line.

## Usage

lein uberjar
java -jar target/chatbot-0.1.0-SNAPSHOT-standalone.jar

## License

Copyright © 2017 By Dmitry Bushenko
