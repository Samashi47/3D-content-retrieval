import { Injectable } from '@angular/core';
import {
  HttpClient,
  HttpErrorResponse,
  HttpHeaders,
} from '@angular/common/http';
import { API_URL } from './env';
import { map, shareReplay } from 'rxjs/operators';
import { Observable } from 'rxjs/internal/Observable';

interface searchResult {
  title: string;
  image: string;
  similarity: number;
}

@Injectable({
  providedIn: 'root',
})
export class SearchService {
  constructor(private http: HttpClient) {}

  downloadModel(filename: string): any {
    const headers = new HttpHeaders()
      .set('Accept', 'model/obj')
      .set('Content-Type', 'application/json');

    return this.http.post(
      `${API_URL}/download-model`,
      { filename },
      {
        headers,
        responseType: 'blob',
      }
    );
  }

  /*search(modelFile: Blob, numberOfResults: number): Observable<searchResult> {
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('numberOfResults', numberOfResults.toString());
    console.log(formData);
    return this.http
      .post<searchResult>(`${API_URL}/search`, formData)
      .pipe(shareReplay());
  }*/
  search(modelFile: Blob, numberOfResults: number): Observable<any> {
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('numberOfResults', numberOfResults.toString());
    console.log(formData);
    return this.http
      .post<any>(`${API_URL}/search`, formData)
      .pipe(shareReplay());
  }
}
