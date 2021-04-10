//Creo el details y el nav automáticamente también
var detalles = document.createElement("details");
detalles.setAttribute("id","indice");
var resumen = document.createElement("summary");
resumen.setAttribute("id", "indice_summary");
resumen.innerText = "Indice";
var navegacion = document.createElement("nav");
navegacion.className = "contenidos";


detalles.appendChild(resumen);
detalles.appendChild(navegacion);
document.body.appendChild(detalles);
console.log(document.getElementsByClassName("contenidos")[0]);
for (var q = 0; q < 9; q++) {
  for (var w = 0; w < document.getElementsByTagName("h"+String(q+1)).length; w++) {
  console.log(document.getElementsByTagName("h"+String(q+1))[w]);}
}

// Creo el índice 
const titulos = document.querySelectorAll("h1, h2, h3, h4, h5, h6"); 

    // h1 sería el titular del documento, lo tratamos por separado
var titular_h1 = document.createElement("h1");
titular_h1.setAttribute("id", "titularh1")
var enlace = document.createElement("a");
enlace.innerHTML = titulos[0].innerHTML;
enlace.href = "#";
titular_h1.appendChild(enlace);
document.getElementsByClassName("contenidos")[0].appendChild(titular_h1);


    // El resto de h2,...,h6 son los distintos apartados
var lista = document.createElement("ul");
document.getElementsByClassName("contenidos")[0].appendChild(lista);
var i = 1;
var j = 0;
var h_actual = 2;
var h_anterior = 2;
var listas = [lista];
while (titulos[i]) {
  var elemento = document.createElement("li");
  var enlace = document.createElement("a");
  enlace.innerHTML = titulos[i].innerHTML;
  elemento.appendChild(enlace);
  lista.appendChild(elemento);
  i++;
  // Al finalizar, añadir lo último y salir:
  if (!titulos[i]) {
    if (h_anterior < h_actual) {
      ultimo.appendChild(lista);}
    break;} 
  // Seguir añadiendo a la misma lista (reiniciando con continue) todos los h iguales
  if (titulos[i].tagName==titulos[i-1].tagName) {
    continue;}
  // Instrucción para todas las iteraciones menos la primera:
  if (j>0 && (h_anterior < h_actual)) {
    ultimo.appendChild(lista);}
  j++;

  var ultimo = elemento;
  var h_actual = Number(titulos[i].tagName[1]);
  var h_anterior = Number(titulos[i-1].tagName[1]);
  // Creamos nueva lista (sublista) o seleccionamos una existente para seguir añadiendo
  if (h_anterior < h_actual) {
    listas[h_anterior-1] = lista;
    var lista = document.createElement("ul");
    continue;}
  if (h_anterior > h_actual) {
    var lista = listas[h_actual-1];
    continue;}
  }


      // listas[0] es la llamada var lista arriba, es la que contiene todo, la de fuera
      // A continuación vamos a poner los href de los <a>

var lista_ppal = listas[0];
lista_ppal.className = "sublista0";
// Apartados y creación de "listas" que utilizaré luego en los subapartados
var selector = ".sublista0 > li";
var elementos = document.querySelectorAll(selector);
var j = 0;
var listas = [];
while (elementos.length >0) {
  for (var i = 0; i<elementos.length; i++) {
    var referencia = "#apartado"+"-"+String(i+1);
    elementos[i].firstChild.href = referencia;
      if (elementos[i].children.length > 1) {
      listas[j] = elementos[i].children[1].children;
      j++;}
  }
  var selector = selector + " > ul > li";
  var elementos = document.querySelectorAll(selector);
}
      // Subapartados
var selector = ".sublista0 > li ul li";
var elementos = document.querySelectorAll(selector);
for (var i = 0; i < elementos.length; i++) {
    var padre_ref = elementos[i].closest("ul").previousSibling.getAttribute("href");
    var j = listas.indexOf(elementos[i].parentElement.children);
    var listas_j = Array.prototype.slice.call(listas[j]); //Paso a Array para poder usar indexOf, que no vale para HTMLCollections
    var k = listas_j.indexOf(elementos[i]);
    var referencia = padre_ref+"-"+String(k+1);
    elementos[i].firstChild.href = referencia;
}


      // Seleccionamos atributo id de los headings iguales a los href de los links

titulos[0].setAttribute("id", "apartado");
var elem_h_anterior = titulos[0];

var contadores = [0,0,0,0,0];

var i = 1;
var j = 0;
while (i < titulos.length) {
  var id = elem_h_anterior.getAttribute("id") + "-" + String(contadores[j]+1);
  titulos[i].setAttribute("id", id);
  contadores[j]+=1;
  i++;
  if (!titulos[i]) {break;} //Para salir tras la última iteracción antes del siguiente if, que me da problemas si no
  if (titulos[i].tagName==titulos[i-1].tagName) {
    continue;}

  var h_actual = Number(titulos[i].tagName[1]);
  var h_anterior = Number(titulos[i-1].tagName[1]);

  //Salida a un heading mayor en la jerarquía (menor en su número)
  if (h_actual < h_anterior) {
    //Reset de todos los contadores de las listas inferiores cuando salgo a una lista superior
    for (var j = h_actual-2+1; j < contadores.length; j++) {
      contadores[j] = 0;} 
    // Selección de la j que corresponde
    var j = h_actual - 2;
    // Rescato el elemento hermano anterior con h# = h(#-1), recorriendo titulos[i] hacia atrás hasta encontrarlo
    for (var k = i; k >= 0; k--) {
      if (Number(titulos[k].tagName[1]) == h_actual - 1) {
        var elem_h_anterior = titulos[k];
        var k = -1}}
    continue;
  }

  else if (h_actual > h_anterior) {
    var j = h_actual - 2;
    var elem_h_anterior = titulos[i-1];
  continue;
  }
}


$("body").click(function(event) {
    if (event.target.id != "indice_summary") {
        detalles.open = false;
    }
});

