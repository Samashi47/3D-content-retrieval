import { Injectable } from '@angular/core';
import {
  HttpClient,
  HttpErrorResponse,
  HttpHeaders,
} from '@angular/common/http';
import { API_URL } from './env';
import { map, shareReplay } from 'rxjs/operators';
import { Observable } from 'rxjs/internal/Observable';

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
}
