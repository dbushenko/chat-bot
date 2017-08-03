(ns chatbot.core
  (:require [clojure.spec.alpha :as s]
            [orchestra.spec.test :as st]
            [expound.alpha :as expound]
            [opennlp.nlp :as nlp]
            [clojure.string :as string])
  (:import opennlp.tools.stemmer.PorterStemmer)
  (:gen-class))

(defn coll-predicate [coll predicate]
  (and (coll? coll) (every? predicate coll)))

(def tokenize (nlp/make-tokenizer "models/en-token.bin"))
(def stemmer (PorterStemmer.))
(def stem (memoize (fn [s] (.stem stemmer s))))

(def ^:const SEPARATOR #"\r\n")

(defrecord QA [^String question
               ^String answer
               tokens])


(defrecord TfIdf [^QA text
                  data])

;;-----------------------

(s/fdef clean-string
        :args (s/cat :str string?)
        :ret string?)

(defn clean-string [^String str]
  (-> str
      (string/replace #"[,\.!\?:;]" "")
      .toLowerCase))

;;-----------------------

(s/fdef str->tokens
        :args (s/cat :s string?)
        :ret #(coll-predicate % string?))

(defn str->tokens [s]
  (->> s
       clean-string
       tokenize
       (map stem)))

;;-----------------------

(s/fdef prepare-text
        :args (s/cat :txt string?)
        :ret #(coll-predicate % (fn [t] (instance? QA t)))
)

(defn prepare-text [txt]
  (let [sp-txt (clojure.string/split txt SEPARATOR)]
    (->> sp-txt
         (partition 2)
         (map (fn [[q a]] (QA. q a (str->tokens q)))))))

;;-----------------------

(s/fdef read-texts
        :ret #(coll-predicate % (fn [t] (instance? QA t))))

(defn read-texts []
  (->> "training"
       clojure.java.io/file
       file-seq
       (filter (memfn isFile))
       (map slurp)
       (map prepare-text)
       (apply concat)))

;;-----------------------

(s/fdef compute-tf
        :args (s/cat :word string?
                     :tokens #(coll-predicate % string?))
        :ret number?)

(defn compute-tf [word tokens]
  (double (/ (count (filter #(= % word) tokens))
             (count tokens))))

(def compute-tf' (memoize compute-tf))

;;-----------------------

;; Computes for one text
(s/fdef tokens->tfs
        :args (s/cat :tokens #(coll-predicate % string?))
        :ret map?)

(defn tokens->tfs [tokens]
  (apply hash-map (apply concat (map #(vector % (compute-tf' % tokens)) tokens))))


;;-----------------------


(s/fdef compute-idf
        :args (s/cat :word string?
                     :tokens-lists (s/and coll?
                                          #(every? coll? %)
                                          #(every? (fn [t] (every? string? t)) %)))
        :ret double?
)

(defn compute-idf [word tokens-lists]
  (let [tokens-sets (map #(into #{} %) tokens-lists)
        docs-num (count tokens-lists)]
    (Math/log (/ docs-num
                 (count (filter #(get % word) tokens-sets))))))


(def compute-idf' (memoize compute-idf))

;;-----------------------

;; Computes for all texts in general
(s/fdef tokens->idfs
        :args (s/cat :tokens #(coll-predicate % coll?))
        :ret map?)

(defn tokens->idfs [tokens-lists]
  (let [words (reduce concat tokens-lists)]
    (apply hash-map
           (apply concat
                  (map #(vector % (compute-idf' % tokens-lists)) words)))))

;;-----------------------

(s/fdef tf-idf
        :args (s/cat :tfs map?   ;; result of tokens->tfs
                     :idfs map?) ;; result of tokens->idfs
        :ret map?)

(defn tf-idf [tfs idfs]
    (apply hash-map
           (apply concat
                  (map #(vector % (double (/ (get tfs %)
                                             (get idfs %))))
                       (keys tfs)))))

;;-----------------------

(s/fdef texts->all-words
        :args (s/cat :texts #(coll-predicate % (fn [t] (instance? QA t))))
        :ret vector?)

(defn texts->all-words [texts]
  (->> texts
       (map :tokens)
       (apply concat)
       (into #{})
       (into [])))

;;-----------------------

(s/fdef words->vec
        :args (s/cat :all-words #(coll-predicate % string?)
                     :words #(coll-predicate % string?)
                     :idfs-map map?)
        :ret #(coll-predicate % double?))

(defn words->vec [all-words words idfs-map]
  (let [tfs-map (tokens->tfs words)]
    (map
     (fn [w] (let [tf (get tfs-map w)
                   idf (get idfs-map w)]
               (if (or (nil? tf)
                       (nil? idf)
                       (= tf 0)
                       (= idf 0))
                 0.0
                 (double (/ tf idf)))))
     all-words)))

;;-----------------------

(s/fdef QA->TfIdf
        :args (s/cat :qas #(coll-predicate % (fn [t] (instance? QA t))))
        :ret #(coll-predicate % (fn [t] (instance? TfIdf t))))

(defn QA->TfIdf [qas]
  (let [idfs (tokens->idfs (map :tokens qas))
        all-words (texts->all-words qas)]
    (map (fn [q] (TfIdf. q (words->vec all-words (:tokens q) idfs)))
         qas)))

;;-----------------------

(s/fdef measure
        :args (s/cat :vec1 #(coll-predicate % double?)
                     :vec2 #(coll-predicate % double?))
        :ret double?)

(defn measure [vec1 vec2]
  (let [nm (reduce + (map * vec1 vec2))
        dnm (* (Math/sqrt (reduce + (map * vec1 vec1)))
               (Math/sqrt (reduce + (map * vec2 vec2))))
       ]
    (Math/abs (double (/ nm dnm)))))

;;-----------------------

(s/fdef find-similar
        :args (s/cat :vc #(coll-predicate % double?)
                     :tf-idfs #(coll-predicate % (fn [t] (instance? TfIdf t))))
        :ret coll?)

(defn find-similar [vc tf-idfs]
  (first (reverse (sort-by first (map #(vector (measure vc (:data %)) %) tf-idfs)))))

;;-----------------------

(defn chat-bot []
  (let [texts (read-texts)
        idfs (tokens->idfs (map :tokens texts))
        all-words (texts->all-words texts)
        tf-idf (QA->TfIdf texts)]
    (loop []
      (let [q (read-line)]
        (if (= q "quit")
          nil
          (let [ts (str->tokens q)
                vc (words->vec all-words ts idfs)
                ans (find-similar vc tf-idf)]
            (println ">>" q)
            (println (-> ans
                         second
                         :text
                         :answer))
            (println "[Confidence:" (* 100 (first ans)) "]")
            (recur)))))))

;;-----------------------

(defn -main [& args]
  (chat-bot))


;;-----------------------


;;(set! s/*explain-out* expound/printer)
;;(st/instrument)

