(defproject chatbot "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha16"]
                 [im.chit/lucid.mind "1.3.13"]
                 [clojure-opennlp "0.4.0"]
                 [orchestra "2017.07.04-1"]
                 [expound "0.1.2"]]
  :main chatbot.core
  :aot :all)
