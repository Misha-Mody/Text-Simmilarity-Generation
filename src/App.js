import React, { useState, useRef } from "react";

const App = () => {
  // first input text
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef(null);

  // second input text
  const [inputValue2, setInputValue2] = useState("");
  const inputRef2 = useRef(null);

  // result from the api
  const [similarityResult, setSimilarityResult] = useState("");
  const [generatedText, setGeneratedText] = useState("");

  // set input values
  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };
  const handleInputChange2 = (event) => {
    setInputValue2(event.target.value);
  };

  // first api to model 1 to calculate similarity between two texts
  const sendDataToAPI = (event) => {
    // get data from the input forms
    event.preventDefault();
    const inputValue = inputRef.current.value;
    const inputValue2 = inputRef2.current.value;

    // model 1 api endpoint
    const apiUrl = "http://127.0.0.1:5000/bert-cosine-similarity";

    // use the input values as payload to the model api
    const payload = {
      texts: [inputValue, inputValue2],
    };

    fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((data) => {
        // get the results
        console.log(data.similarity_matrix);
        setSimilarityResult(
          (Math.round(data.similarity_matrix[0][1] * 100) / 100) * 100
        );
        setInputValue("");
        setInputValue2("");
      })
      .catch((error) => {
        console.error(error);
      });
  };

  // second api to model 2 to generate simialt text
  const generateDataAPI = (event) => {
    // get data from the input forms
    event.preventDefault();
    const inputValue = inputRef.current.value;
    const inputValue2 = inputRef2.current.value;
    const apiUrl = "http://127.0.0.1:5000/generate-text";
    const payload = {
      texts: [inputValue, inputValue2],
    };
    fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((data) => {
        setGeneratedText(data.generated_text);
        setInputValue("");
        setInputValue2("");
      })
      .catch((error) => {
        console.error(error);
      });
  };

  return (
    <div className="container">
      <div className="row">
        <h1 className="text-center mt-5">React.js App</h1>

        {/* Two Input Fields */}
        <form className="mt-5">
          <div className="form-group">
            <div className="row">
              <div className="col-lg-6 col-sm-12">
                <label htmlFor="text1">
                  <b>INPUT TEXT 1:</b>
                </label>
                <br />
                <input
                  value={inputValue}
                  onChange={handleInputChange}
                  type="text"
                  className="form-control form-control-lg"
                  id="text1"
                  ref={inputRef}
                />
              </div>
              <div className="col-lg-6 col-sm-12">
                <label htmlFor="text2">
                  <b>INPUT TEXT 2:</b>
                </label>
                <br />
                <input
                  value={inputValue2}
                  onChange={handleInputChange2}
                  ref={inputRef2}
                  className="form-control form-control-lg"
                  type="text"
                  id="text2"
                />
              </div>
            </div>
          </div>
        </form>

        {/* Text Similarity Block */}
        <div className="row  mt-5" style={{ display: "inline-block" }}>
          <button
            style={{ padding: "5px 20px", width: "auto" }}
            className="btn btn-primary"
            onClick={sendDataToAPI}
          >
            Find Similarity Between the Two Texts
          </button>
          &nbsp;&nbsp;&nbsp;The Similarity is:&nbsp;
          {similarityResult && similarityResult + "%"}
        </div>

        {/* Generate Data Block */}
        <div className="row  mt-5">
          <button
            style={{ padding: "5px 20px", width: "auto" }}
            className="btn btn-primary mb-3"
            onClick={generateDataAPI}
          >
            Generate New Text
          </button>
          <p>{generatedText && generatedText}</p>
        </div>
      </div>
    </div>
  );
};

export default App;
